from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ...shared.coarse_graph import E_SU, NUM_SU_TYPES, PPM_AXIS, get_su_ppm_range
from ...shared.inverse_common import (
    _NodeV3,
    evaluate_spectrum_reconstruction,
    lorentzian_spectrum,
)


class Layer3Estimator:
    """Optimize z inside each node's fixed Layer2 template."""

    def __init__(self, device: torch.device, vae_model=None, lib_path: Optional[str] = None):
        self.device = device
        self.vae = vae_model
        self.lib_path = lib_path
        self._template_cache: Optional[Dict] = None
        self.last_diagnostics: Dict[str, object] = {}

    def _get_template_library(self) -> Optional[Dict]:
        if self._template_cache is not None:
            return self._template_cache
        if not self.lib_path:
            return None
        path = Path(self.lib_path)
        if not path.exists():
            return None
        try:
            try:
                lib = torch.load(str(path), map_location='cpu', weights_only=True)
            except (TypeError, RuntimeError):
                lib = torch.load(str(path), map_location='cpu', weights_only=False)
        except Exception as exc:
            import logging
            logging.warning(f"Failed to load template library from {path}: {exc}")
            return None
        if not isinstance(lib, dict):
            return None
        self._template_cache = {'templates': lib.get('templates', {})}
        return self._template_cache

    def _global_embed_from_elements(self, E_target: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("Layer3 requires a G2S model")
        e = E_target.to(self.device).view(1, -1)
        if float(e.max().item()) > 1.1:
            e = e / e.sum(dim=1, keepdim=True).clamp(min=1.0)
        try:
            with torch.inference_mode():
                embedding = 0.02 * self.vae.global_mlp(e)
        except Exception as exc:
            raise RuntimeError(f"Layer3 failed to compute the G2S global embedding: {exc}") from exc
        if embedding.ndim != 2 or int(embedding.shape[0]) != 1 or not bool(torch.isfinite(embedding).all()):
            raise RuntimeError(
                f"Layer3 received an invalid G2S global embedding with shape={tuple(embedding.shape)}"
            )
        return embedding

    def _decode_mu_batch(self,
                         center_sus: List[int],
                         z_vectors: List[torch.Tensor],
                         g_embed: torch.Tensor,
                         context: str) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("Layer3 requires a G2S model")
        if len(center_sus) != len(z_vectors):
            raise RuntimeError(
                f"Layer3 {context} decoder input mismatch: "
                f"center_sus={len(center_sus)}, z_vectors={len(z_vectors)}"
            )
        if not z_vectors:
            return torch.empty(0, dtype=torch.float, device=self.device)

        normalized_z: List[torch.Tensor] = []
        latent_dim: Optional[int] = None
        for index, z_vec in enumerate(z_vectors):
            if not torch.is_tensor(z_vec) or int(z_vec.numel()) <= 0:
                raise RuntimeError(f"Layer3 {context} has missing z at index={index}")
            z_flat = z_vec.detach().to(self.device).float().reshape(-1)
            if not bool(torch.isfinite(z_flat).all()):
                raise RuntimeError(f"Layer3 {context} has non-finite z at index={index}")
            if latent_dim is None:
                latent_dim = int(z_flat.numel())
            elif int(z_flat.numel()) != int(latent_dim):
                raise RuntimeError(
                    f"Layer3 {context} has inconsistent z dimensions: "
                    f"expected={latent_dim}, actual={int(z_flat.numel())}, index={index}"
                )
            normalized_z.append(z_flat)

        z_batch = torch.stack(normalized_z, dim=0)
        su_index = torch.tensor(center_sus, dtype=torch.long, device=self.device)
        if bool(((su_index < 0) | (su_index >= NUM_SU_TYPES)).any()):
            raise RuntimeError(f"Layer3 {context} contains an invalid center SU")
        su_feat = F.one_hot(su_index, num_classes=NUM_SU_TYPES).float()
        if int(g_embed.shape[0]) == 1:
            global_batch = g_embed.expand(int(z_batch.shape[0]), -1)
        elif int(g_embed.shape[0]) == int(z_batch.shape[0]):
            global_batch = g_embed
        else:
            raise RuntimeError(
                f"Layer3 {context} global embedding batch mismatch: "
                f"g={int(g_embed.shape[0])}, z={int(z_batch.shape[0])}"
            )

        try:
            with torch.inference_mode():
                pred = self.vae.decoder(su_feat, z_batch, global_batch)
        except Exception as exc:
            raise RuntimeError(f"Layer3 {context} G2S decoding failed: {exc}") from exc
        if pred.ndim != 2 or int(pred.shape[0]) != len(z_vectors) or int(pred.shape[1]) < 1:
            raise RuntimeError(
                f"Layer3 {context} G2S decoder returned invalid shape={tuple(pred.shape)}"
            )
        mu = pred[:, 0].detach()
        if not bool(torch.isfinite(mu).all()):
            raise RuntimeError(f"Layer3 {context} G2S decoder returned non-finite chemical shifts")
        return mu

    def _decode_nodes(self,
                      carbon_nodes: List[_NodeV3],
                      g_embed: torch.Tensor,
                      context: str) -> torch.Tensor:
        return self._decode_mu_batch(
            [int(node.su_type) for node in carbon_nodes],
            [getattr(node, 'z_vec', None) for node in carbon_nodes],
            g_embed,
            context=context,
        )

    @staticmethod
    def _segment_mask(ppm_axis: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return (ppm_axis >= float(lo)) & (ppm_axis <= float(hi))

    @staticmethod
    def _segment_abs(diff: torch.Tensor, mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return float('inf')
        return float(torch.sum(torch.abs(diff[mask])))

    def _top_negative_ppms(self,
                           diff: torch.Tensor,
                           ppm_axis: torch.Tensor,
                           mask: torch.Tensor,
                           top_k: int,
                           min_sep_ppm: float) -> List[float]:
        idxs = torch.nonzero(mask & (diff < 0), as_tuple=False).view(-1)
        if int(idxs.numel()) <= 0:
            return []
        order = torch.argsort(diff[idxs])
        selected: List[float] = []
        for order_idx in order:
            ppm = float(ppm_axis[int(idxs[int(order_idx)])])
            if any(abs(ppm - old) < float(min_sep_ppm) for old in selected):
                continue
            selected.append(ppm)
            if len(selected) >= max(1, int(top_k)):
                break
        return selected

    def _best_positive_in_range(self,
                                diff: torch.Tensor,
                                ppm_axis: torch.Tensor,
                                lo: float,
                                hi: float) -> Optional[Tuple[float, float]]:
        mask = self._segment_mask(ppm_axis, lo, hi)
        idxs = torch.nonzero(mask, as_tuple=False).view(-1)
        if int(idxs.numel()) <= 0:
            return None
        values = diff[idxs]
        value, local_idx = torch.max(values, dim=0)
        if float(value) <= 0.0:
            return None
        axis_idx = int(idxs[int(local_idx)])
        return float(ppm_axis[axis_idx]), float(value)

    @staticmethod
    def _mu_sorted_candidates(mu_samples: torch.Tensor,
                              target_mu: float,
                              top_k: int) -> List[int]:
        if not torch.is_tensor(mu_samples) or int(mu_samples.numel()) <= 0:
            return []
        distances = np.abs(mu_samples.detach().cpu().numpy().astype(np.float64) - float(target_mu))
        return [int(idx) for idx in np.argsort(distances)[:max(1, int(top_k))]]

    @staticmethod
    def _peak(mu: float, ppm_axis: torch.Tensor) -> torch.Tensor:
        return lorentzian_spectrum(
            torch.tensor([float(mu)], dtype=torch.float, device=ppm_axis.device),
            torch.ones(1, dtype=torch.float, device=ppm_axis.device),
            ppm_axis,
            hwhm=1.0,
        )

    def _build_recon(self,
                     nodes: List[_NodeV3],
                     ppm_axis: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        peak_cache: Dict[int, torch.Tensor] = {}
        recon = torch.zeros_like(ppm_axis)
        for node in nodes:
            su_type = int(node.su_type)
            if float(E_SU[su_type, 0].detach().cpu().item()) <= 0.0:
                continue
            mu = float(getattr(node, 'mu', 0.0))
            if not np.isfinite(mu) or mu <= 0.0:
                continue
            peak = self._peak(mu, ppm_axis)
            peak_cache[int(node.global_id)] = peak
            recon = recon + peak
        return recon, peak_cache

    def layer3_adjust_templates(
        self,
        nodes: List[_NodeV3],
        S_target: torch.Tensor,
        E_target: torch.Tensor,
        max_iters: int = 150,
        output_dir: Optional[str] = None,
        pos_search_window: float = 15.0,
        neg_assign_window: float = 2.0,
        top_k_samples: int = 8,
        neg_top_k_peaks: int = 10,
        neg_peak_min_sep_ppm: float = 0.8,
        verbose: bool = True,
    ) -> Tuple[List[_NodeV3], float]:
        lib = self._get_template_library()
        templates = (lib or {}).get('templates', {}) if isinstance(lib, dict) else {}
        if not templates:
            raise RuntimeError("Layer3 template library is unavailable or empty")
        if self.vae is None:
            raise RuntimeError("Layer3 requires G2S decoding for every z candidate")
        if hasattr(self.vae, 'eval'):
            self.vae.eval()

        ppm_axis = PPM_AXIS.to(self.device)
        target = S_target.to(self.device).view(-1)
        baseline_nodes = deepcopy(nodes)
        baseline_has_complete_peaks = all(
            np.isfinite(float(getattr(node, 'mu', 0.0)))
            and float(getattr(node, 'mu', 0.0)) > 0.0
            for node in baseline_nodes
            if float(E_SU[int(node.su_type), 0].detach().cpu().item()) > 0.0
        )
        baseline_recon, _ = self._build_recon(baseline_nodes, ppm_axis)
        baseline_eval = evaluate_spectrum_reconstruction(
            target,
            baseline_recon,
            ppm_axis=ppm_axis,
        )
        baseline_r2 = float(baseline_eval.get('r2', float('-inf')))
        g_embed = self._global_embed_from_elements(E_target)
        carbon_nodes: List[_NodeV3] = []
        fixed_template_keys: Dict[int, Tuple] = {}
        unique_template_keys: Dict[Tuple, int] = {}
        coverage_errors: List[str] = []

        for node in nodes:
            su_type = int(node.su_type)
            if float(E_SU[su_type, 0].detach().cpu().item()) <= 0.0:
                continue
            carbon_nodes.append(node)
            template_key = getattr(node, 'template_key', None)
            template = templates.get(template_key) if isinstance(template_key, tuple) else None
            if not isinstance(template, dict):
                coverage_errors.append(f"node={int(node.global_id)} has no fixed Layer2 template")
                continue
            samples = template.get('samples', {}) or {}
            z_samples = samples.get('z')
            if not torch.is_tensor(z_samples) or int(z_samples.numel()) <= 0:
                coverage_errors.append(
                    f"node={int(node.global_id)} template={template_key} has no z sample bank"
                )
                continue
            if z_samples.ndim == 1:
                z_samples = z_samples.view(1, -1)
            if z_samples.ndim != 2 or int(z_samples.shape[0]) <= 0:
                coverage_errors.append(
                    f"node={int(node.global_id)} template={template_key} has invalid "
                    f"z sample shape={tuple(z_samples.shape)}"
                )
                continue
            fixed_template_keys[int(node.global_id)] = template_key
            unique_template_keys[template_key] = int(su_type)
            node.pi = 1.0

        if coverage_errors:
            raise RuntimeError("Layer3 fixed-template coverage failed: " + " | ".join(coverage_errors[:8]))

        decoded_template_mu: Dict[Tuple, torch.Tensor] = {}
        for template_key, center_su in sorted(unique_template_keys.items(), key=lambda item: repr(item[0])):
            template = templates[template_key]
            z_samples = template['samples']['z']
            if z_samples.ndim == 1:
                z_samples = z_samples.view(1, -1)
            decoded_template_mu[template_key] = self._decode_mu_batch(
                [int(center_su)] * int(z_samples.shape[0]),
                [z_samples[index] for index in range(int(z_samples.shape[0]))],
                g_embed,
                context=f"template={template_key} sample bank",
            )

        entry_mu = self._decode_nodes(carbon_nodes, g_embed, context="entry all-node decode")
        for node, mu_tensor in zip(carbon_nodes, entry_mu):
            mu_value = float(mu_tensor.item())
            node.mu = mu_value
            node.pi = 1.0
            score_components = getattr(node, 'score_components', None)
            if isinstance(score_components, dict):
                score_components['layer3_spectral_source'] = 'g2s_layer3'
                score_components['spectral_source'] = 'g2s_layer3'
                score_components['layer3_decoder_mu'] = mu_value

        recon, peak_cache = self._build_recon(nodes, ppm_axis)
        eval_current = evaluate_spectrum_reconstruction(
            target,
            recon,
            ppm_axis=ppm_axis,
        )
        current_r2 = float(eval_current.get('r2', 0.0))
        full_mask = torch.ones_like(ppm_axis, dtype=torch.bool)
        moves: List[Dict] = []
        stop_reason = 'max_iters'

        if verbose:
            print("\n" + "=" * 60)
            print("Layer3: 固定模板潜变量 z 优化")
            print("=" * 60)
            print(
                f"  [Layer3 start] nodes={len(nodes)}, carbon_nodes={len(carbon_nodes)}, "
                f"max_iters={int(max_iters)}, initial_r2={current_r2:.4f}"
            )

        for _ in range(max(0, int(max_iters))):
            diff = eval_current['S_target'] - eval_current['S_recon']
            negative_ppms = self._top_negative_ppms(
                diff,
                ppm_axis,
                full_mask,
                top_k=int(neg_top_k_peaks),
                min_sep_ppm=float(neg_peak_min_sep_ppm),
            )
            if not negative_ppms:
                stop_reason = 'no_negative_peaks'
                break

            best_move: Optional[Dict] = None
            for negative_ppm in negative_ppms:
                candidate_nodes = [
                    node for node in carbon_nodes
                    if abs(float(node.mu) - float(negative_ppm)) <= float(neg_assign_window)
                ]
                candidate_nodes.sort(
                    key=lambda node: (abs(float(node.mu) - float(negative_ppm)), int(node.global_id))
                )
                for node in candidate_nodes:
                    template_key = fixed_template_keys[int(node.global_id)]
                    template = templates[template_key]
                    samples = template['samples']
                    z_samples = samples['z']
                    if z_samples.ndim == 1:
                        z_samples = z_samples.view(1, -1)
                    decoded_mu_samples = decoded_template_mu[template_key]
                    coarse_lo, coarse_hi = get_su_ppm_range(int(node.su_type))
                    search_lo = max(float(coarse_lo), float(negative_ppm) - float(pos_search_window))
                    search_hi = min(float(coarse_hi), float(negative_ppm) + float(pos_search_window))
                    if search_hi <= search_lo:
                        continue
                    positive = self._best_positive_in_range(diff, ppm_axis, search_lo, search_hi)
                    if positive is None:
                        continue
                    positive_ppm, positive_value = positive
                    sample_indices = self._mu_sorted_candidates(
                        decoded_mu_samples,
                        positive_ppm,
                        top_k=int(top_k_samples),
                    )
                    old_peak = peak_cache.get(int(node.global_id))
                    if old_peak is None:
                        continue

                    for sample_idx in sample_indices:
                        z_candidate = z_samples[int(sample_idx)].detach().clone().to(self.device)
                        old_z = getattr(node, 'z_vec', None)
                        if (
                            torch.is_tensor(old_z)
                            and int(old_z.numel()) == int(z_candidate.numel())
                            and torch.equal(
                                old_z.detach().to(self.device).reshape(-1),
                                z_candidate.reshape(-1),
                            )
                        ):
                            continue
                        mu_candidate = float(decoded_mu_samples[int(sample_idx)].item())
                        if not (float(coarse_lo) <= float(mu_candidate) <= float(coarse_hi)):
                            continue

                        peak_candidate = self._peak(mu_candidate, ppm_axis)
                        recon_candidate = recon - old_peak + peak_candidate
                        eval_candidate = evaluate_spectrum_reconstruction(
                            target,
                            recon_candidate,
                            ppm_axis=ppm_axis,
                        )
                        r2_candidate = float(eval_candidate.get('r2', 0.0))
                        if r2_candidate <= current_r2 + 1e-9:
                            continue
                        residual_candidate = self._segment_abs(
                            eval_candidate['S_target'] - eval_candidate['S_recon'],
                            full_mask,
                        )
                        rank = (r2_candidate, -residual_candidate)
                        if best_move is None or rank > best_move['rank']:
                            best_move = {
                                'rank': rank,
                                'node': node,
                                'template_key': template_key,
                                'z': z_candidate,
                                'mu': float(mu_candidate),
                                'peak': peak_candidate,
                                'recon': recon_candidate,
                                'eval': eval_candidate,
                                'sample_idx': int(sample_idx),
                                'negative_ppm': float(negative_ppm),
                                'positive_ppm': float(positive_ppm),
                                'positive_value': float(positive_value),
                                'coarse_lo': float(coarse_lo),
                                'coarse_hi': float(coarse_hi),
                            }

            if best_move is None:
                stop_reason = 'no_r2_improving_z_move'
                break

            node = best_move['node']
            old_mu = float(node.mu)
            old_r2 = float(current_r2)
            node.z_vec = best_move['z']
            node.mu = float(best_move['mu'])
            node.pi = 1.0
            score_components = getattr(node, 'score_components', None)
            if isinstance(score_components, dict):
                score_components['layer3_sample_idx'] = int(best_move['sample_idx'])
                score_components['layer3_spectral_source'] = 'g2s_layer3'
                score_components['spectral_source'] = 'g2s_layer3'
                score_components['layer3_decoder_mu'] = float(best_move['mu'])
            if isinstance(getattr(node, 'z_history', None), list):
                node.z_history.append(best_move['z'].detach().clone())

            peak_cache[int(node.global_id)] = best_move['peak']
            recon = best_move['recon']
            eval_current = best_move['eval']
            current_r2 = float(eval_current.get('r2', 0.0))
            moves.append({
                'move_type': 'fixed_template_z',
                'global_id': int(node.global_id),
                'su_type': int(node.su_type),
                'template_key': str(best_move['template_key']),
                'sample_idx': int(best_move['sample_idx']),
                'negative_ppm': float(best_move['negative_ppm']),
                'positive_ppm': float(best_move['positive_ppm']),
                'positive_value': float(best_move['positive_value']),
                'coarse_lo': float(best_move['coarse_lo']),
                'coarse_hi': float(best_move['coarse_hi']),
                'old_mu': old_mu,
                'new_mu': float(best_move['mu']),
                'r2_before': old_r2,
                'r2_after': current_r2,
            })

        for node in carbon_nodes:
            if getattr(node, 'template_key', None) != fixed_template_keys[int(node.global_id)]:
                raise RuntimeError(f"Layer3 changed fixed template for node={int(node.global_id)}")

        exit_mu = self._decode_nodes(carbon_nodes, g_embed, context="exit all-node decode")
        max_decode_error = 0.0
        for node, mu_tensor in zip(carbon_nodes, exit_mu):
            decoded_mu = float(mu_tensor.item())
            stored_mu = float(getattr(node, 'mu', float('nan')))
            error = abs(stored_mu - decoded_mu)
            max_decode_error = max(max_decode_error, error)
            if not np.isfinite(error) or error > 1e-5:
                raise RuntimeError(
                    f"Layer3 final G2S consistency failed for node={int(node.global_id)}: "
                    f"stored_mu={stored_mu}, decoded_mu={decoded_mu}, error={error}"
                )
            node.mu = decoded_mu
            node.pi = 1.0
            score_components = getattr(node, 'score_components', None)
            if isinstance(score_components, dict):
                score_components['layer3_spectral_source'] = 'g2s_layer3'
                score_components['spectral_source'] = 'g2s_layer3'
                score_components['layer3_decoder_mu'] = decoded_mu
                score_components['layer3_final_decode_error'] = float(error)

        recon, peak_cache = self._build_recon(nodes, ppm_axis)
        eval_current = evaluate_spectrum_reconstruction(
            target,
            recon,
            ppm_axis=ppm_axis,
        )
        spectrum_recon = eval_current['S_recon']
        spectrum_diff = eval_current['S_target'] - spectrum_recon
        r2 = float(eval_current.get('r2', 0.0))
        accepted = bool(
            np.isfinite(r2)
            and (
                not baseline_has_complete_peaks
                or r2 + 1e-9 >= baseline_r2
            )
        )
        self.last_diagnostics = {
            'accepted': bool(accepted),
            'reason': (
                'g2s_layer3_required_complete_decode'
                if accepted and not baseline_has_complete_peaks
                else 'g2s_layer3_not_worse_than_input'
                if accepted
                else 'g2s_layer3_below_input_baseline'
            ),
            'baseline_r2': float(baseline_r2),
            'decoded_initial_r2': float(current_r2 if not moves else moves[0]['r2_before']),
            'candidate_r2': float(r2),
            'r2_gain': float(r2 - baseline_r2),
            'z_moves': int(len(moves)),
            'stop_reason': str(stop_reason),
            'max_decode_error': float(max_decode_error),
        }

        if not accepted:
            print(
                f"  [Layer3 rollback] candidate_r2={r2:.4f} < "
                f"input_r2={baseline_r2:.4f}; retained incoming Layer2 state"
            )
            print("Layer3 完成（候选未提交）\n")
            return baseline_nodes, float(baseline_r2)

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                'ppm': ppm_axis.detach().cpu().numpy(),
                'target': target.detach().cpu().numpy(),
                'reconstructed': spectrum_recon.detach().cpu().numpy(),
                'difference': spectrum_diff.detach().cpu().numpy(),
            }).to_csv(str(out / 'layer3_spectrum.csv'), index=False)
            pd.DataFrame(moves).to_csv(str(out / 'layer3_z_moves.csv'), index=False)

        print(
            f"  最终调整结果: z_moves={len(moves)}, template_moves=0, "
            f"stop={stop_reason}, max_decode_error={max_decode_error:.2e}, "
            f"R²={r2:.4f}"
        )
        print("Layer3 完成\n")
        return nodes, r2
