import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .coarse_graph import NUM_SU_TYPES, PPM_AXIS, E_SU
from .inverse_common import _NodeV3, lorentzian_spectrum, evaluate_spectrum_reconstruction

class Layer3Estimator:
    def __init__(self, device: torch.device, vae_model=None, lib_path: Optional[str] = None):
        self.device = device
        self.vae = vae_model
        self.lib_path = lib_path
        self._template_cache: Optional[Dict] = None
        self.intensity_scale = 1.0

    def _get_template_library(self) -> Optional[Dict]:
        if self._template_cache is not None:
            return self._template_cache
        if not self.lib_path:
            return None
        p = Path(self.lib_path)
        if not p.exists():
            return None
        try:
            try:
                lib = torch.load(str(p), map_location='cpu', weights_only=True)
            except TypeError:
                lib = torch.load(str(p), map_location='cpu')
            except Exception:
                lib = torch.load(str(p), map_location='cpu')
        except Exception as e:
            import logging
            logging.warning(f"Failed to load template library from {p}: {e}")
            return None
        if not isinstance(lib, dict):
            return None
        self._template_cache = {
            'templates': lib.get('templates', {}),
            'center_index': lib.get('center_index', {}),
            'role_index': lib.get('role_index', {}),
        }
        return self._template_cache

    def _global_embed_from_elements(self, E_target: torch.Tensor) -> torch.Tensor:
        device = self.device
        e = E_target.to(device).view(1, -1)
        if float(e.max().item()) > 1.1:
            s = e.sum(dim=1, keepdim=True).clamp(min=1.0)
            e = e / s
        if self.vae is not None:
            try:
                g = self.vae.global_mlp(e)
                return 0.02 * g
            except Exception as e:
                import logging
                logging.warning(f"Failed to compute global embedding: {e}")
        return torch.zeros((1, 2), dtype=torch.float, device=device)

    def _decode_mu_pi_from_z(self, center_su: int, z_vec: torch.Tensor, g_embed: torch.Tensor) -> Optional[Tuple[float, float]]:
        if self.vae is None:
            return None
        device = self.device
        try:
            su_feat = F.one_hot(
                torch.tensor([int(center_su)], dtype=torch.long, device=device),
                num_classes=NUM_SU_TYPES,
            ).float()
            z = z_vec.to(device).view(1, -1)
            pred = self.vae.decoder(su_feat, z, g_embed)
            mu_pred = float(pred[:, 0].detach().item())
            pi_pred = float(F.softplus(pred[:, 1]).detach().item())
            return mu_pred, pi_pred
        except Exception:
            return None

    def _segment_mask(self, ppm_axis: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return (ppm_axis >= float(lo)) & (ppm_axis <= float(hi))

    def _segment_abs(self, diff: torch.Tensor, mask: torch.Tensor) -> float:
        # Optimized to avoid item() sync on GPU if diff is on CPU
        if not bool(mask.any()):
            return float('inf')
        return float(torch.sum(torch.abs(diff[mask])))

    def _max_negative_ppm(self, diff: torch.Tensor, ppm_axis: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
        if not bool(mask.any()):
            return None
        idxs = torch.nonzero(mask, as_tuple=False).view(-1)
        seg = diff[idxs]
        if seg.numel() <= 0:
            return None
        min_val, min_idx = torch.min(seg, dim=0)
        if float(min_val) >= 0.0:
            return None
        i = int(idxs[min_idx])
        return float(ppm_axis[i])

    def _top_negative_ppms(self,
                           diff: torch.Tensor,
                           ppm_axis: torch.Tensor,
                           mask: torch.Tensor,
                           top_k: int,
                           min_sep_ppm: float) -> List[float]:
        if not bool(mask.any()):
            return []
        idxs = torch.nonzero(mask, as_tuple=False).view(-1)
        seg = diff[idxs]
        if seg.numel() <= 0:
            return []

        neg_mask = seg < 0
        if not bool(neg_mask.any()):
            return []

        neg_idxs = idxs[neg_mask]
        neg_vals = seg[neg_mask]
        order = torch.argsort(neg_vals)  # more negative first

        selected: List[float] = []
        k = max(1, int(top_k))
        sep = float(min_sep_ppm)

        for j in order:
            try:
                base_idx = int(neg_idxs[j])
            except Exception:
                continue
            ppm = float(ppm_axis[base_idx])
            if sep > 0 and any(abs(ppm - p0) < sep for p0 in selected):
                continue
            selected.append(ppm)
            if len(selected) >= k:
                break
        return selected

    def _best_positive_in_range(self, diff: torch.Tensor, ppm_axis: torch.Tensor, lo: float, hi: float) -> Optional[Tuple[float, float]]:
        m = self._segment_mask(ppm_axis, lo, hi)
        if not bool(m.any()):
            return None
        idxs = torch.nonzero(m, as_tuple=False).view(-1)
        seg = diff[idxs]
        if seg.numel() <= 0:
            return None
        vmax, max_idx = torch.max(seg, dim=0)
        if float(vmax) <= 0.0:
            return None
        i = int(idxs[max_idx])
        return float(ppm_axis[i]), float(vmax)

    def _mu_sorted_candidates(self, mu_samples: torch.Tensor, target_mu: float, top_k: int) -> List[int]:
        if mu_samples is None or int(mu_samples.numel()) <= 0:
            return []
        mu = mu_samples.detach().cpu().numpy().astype(np.float64)
        d = np.abs(mu - float(target_mu))
        order = np.argsort(d)
        out: List[int] = []
        for idx in order[: max(1, int(top_k))]:
            out.append(int(idx))
        return out

    def _peak(self, mu: float, pi: float, ppm_axis: torch.Tensor, hwhm: float) -> torch.Tensor:
        mus = torch.tensor([float(mu)], dtype=torch.float, device=ppm_axis.device)
        pis = torch.tensor([float(pi)], dtype=torch.float, device=ppm_axis.device)
        return lorentzian_spectrum(mus, pis, ppm_axis, hwhm=float(hwhm))

    def _build_recon(self, nodes: List[_NodeV3], ppm_axis: torch.Tensor, hwhm: float) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        peak_cache: Dict[int, torch.Tensor] = {}
        recon = torch.zeros_like(ppm_axis)
        try:
            s = float(getattr(self, 'intensity_scale', 1.0))
        except Exception as e:
            import logging
            logging.debug(f"Failed to get intensity_scale: {e}")
            s = 1.0
        for n in nodes:
            su = int(n.su_type)
            try:
                is_carbon = float(E_SU[int(su), 0].detach().cpu().item()) > 0.0
            except Exception as e:
                import logging
                logging.debug(f"Failed to check if SU is carbon: {e}")
                is_carbon = False
            if not bool(is_carbon):
                continue
            mu = float(getattr(n, 'mu', 0.0))
            pi = float(getattr(n, 'pi', 0.0))
            if float(mu) == 0.0 or float(pi) <= 0.0:
                continue
            pi_eff = float(pi) * float(s)
            pk = self._peak(mu, pi_eff, ppm_axis, hwhm=float(hwhm))
            peak_cache[int(n.global_id)] = pk
            recon = recon + pk
        return recon, peak_cache

    def _multiset_from_counter(self, cnt: Counter) -> Tuple[int, ...]:
        ms: List[int] = []
        if cnt is None:
            return tuple()
        for k, v in cnt.items():
            try:
                kk = int(k)
                vv = int(v)
            except Exception:
                continue
            if vv <= 0:
                continue
            ms.extend([kk] * vv)
        ms.sort()
        return tuple(ms)

    def _multiset_overlap_size(self, ms1: Tuple[int, ...], ms2: Tuple[int, ...]) -> int:
        c1 = Counter(ms1)
        c2 = Counter(ms2)
        all_keys = set(c1.keys()) | set(c2.keys())
        return int(sum(min(int(c1.get(k, 0)), int(c2.get(k, 0))) for k in all_keys))

    def _multiset_diff_nodes(self, ms1: Tuple[int, ...], ms2: Tuple[int, ...]) -> int:
        ov = self._multiset_overlap_size(ms1, ms2)
        return int(max(len(ms1), len(ms2)) - int(ov))

    def _is_h2_approx_node(self, n: _NodeV3) -> bool:
        tpl_key = getattr(n, 'template_key', None)
        if not (isinstance(tpl_key, tuple) and len(tpl_key) == 3):
            return False
        try:
            hop1_obs = self._multiset_from_counter(getattr(n, 'hop1_su', Counter()) or Counter())
            hop2_obs = self._multiset_from_counter(getattr(n, 'hop2_su', Counter()) or Counter())
        except Exception as e:
            import logging
            logging.debug(f"Failed to get node hop multisets: {e}")
            return False
        try:
            tpl_h1 = tuple(tpl_key[1])
            tpl_h2 = tuple(tpl_key[2])
        except Exception as e:
            import logging
            logging.debug(f"Invalid template key format: {e}")
            return False
        if tuple(tpl_h1) != tuple(hop1_obs):
            return False
        return tuple(tpl_h2) != tuple(hop2_obs)

    def layer3_adjust_templates(
        self,
        nodes: List[_NodeV3],
        S_target: torch.Tensor,
        E_target: torch.Tensor,
        max_iters: int = 30,
        hwhm: float = 1.0,
        output_dir: Optional[str] = None,
        pos_search_window: float = 10.0,
        neg_assign_window: float = 1.5,
        top_k_samples: int = 5,
        neg_top_k_peaks: int = 10,
        neg_peak_min_sep_ppm: float = 0.8,
        enable_approx_hop2_template_adjust: bool = False,
        approx_hop2_max_iters: Optional[int] = None,
        approx_hop2_max_diff_nodes: int = 3,
        approx_hop2_top_k_templates: int = 80,
    ) -> Tuple[List[_NodeV3], float]:
        lib = self._get_template_library()
        templates = (lib or {}).get('templates', {}) if isinstance(lib, dict) else {}

        device = self.device
        ppm_axis = PPM_AXIS.to(device)
        S_t = S_target.to(device).view(-1)

        g_embed = self._global_embed_from_elements(E_target)

        recon, peak_cache = self._build_recon(nodes, ppm_axis, hwhm=float(hwhm))
        moves: List[Dict] = []
        
        print("\n" + "=" * 60)
        print("Layer3: 潜变量z与模板优化")
        print("=" * 60)

        regions = [
            (0.0, 90.0, [22, 23, 24, 25, 19, 20, 21]),
            (90.0, 160.0, [13, 12, 11, 10, 9, 5, 6, 7, 8, 14, 15, 16, 17, 18, 4]),
            (160.0, 240.0, [1, 2, 3, 0]),
        ]

        for region_lo, region_hi, pri_su in regions:
            for _ in range(max(1, int(max_iters))):
                diff = S_t - recon
                region_mask = self._segment_mask(ppm_axis, float(region_lo), float(region_hi))
                loss_before = self._segment_abs(diff, region_mask)
                neg_ppm_candidates = self._top_negative_ppms(
                    diff=diff,
                    ppm_axis=ppm_axis,
                    mask=region_mask,
                    top_k=int(neg_top_k_peaks),
                    min_sep_ppm=float(neg_peak_min_sep_ppm),
                )
                if not neg_ppm_candidates:
                    break

                accepted = False
                for neg_ppm in neg_ppm_candidates:
                    for su_type in pri_su:
                        cand_nodes = [
                            n for n in nodes
                            if int(n.su_type) == int(su_type)
                            and abs(float(getattr(n, 'mu', 0.0)) - float(neg_ppm)) <= float(neg_assign_window)
                        ]
                        cand_nodes.sort(key=lambda n: int(n.global_id))

                        for n in cand_nodes:
                            tpl_key = getattr(n, 'template_key', None)
                            if tpl_key is None:
                                continue
                            tpl = templates.get(tpl_key)
                            if not isinstance(tpl, dict):
                                continue
                            samples = tpl.get('samples', {})
                            if not isinstance(samples, dict):
                                continue
                            z_samples = samples.get('z', None)
                            mu_samples = samples.get('mu', None)
                            pi_samples = samples.get('pi', None)
                            if z_samples is None or mu_samples is None or pi_samples is None:
                                continue
                            if int(mu_samples.numel()) <= 0:
                                continue

                            mu_min = float(tpl.get('mu_min', 0.0))
                            mu_max = float(tpl.get('mu_max', 0.0))
                            lo = max(float(region_lo), float(mu_min), float(neg_ppm) - float(pos_search_window))
                            hi = min(float(region_hi), float(mu_max), float(neg_ppm) + float(pos_search_window))
                            if hi <= lo:
                                continue

                            best_pos = self._best_positive_in_range(diff, ppm_axis, lo, hi)
                            if best_pos is None:
                                continue
                            pos_ppm, pos_val = best_pos

                            cand_idx = self._mu_sorted_candidates(mu_samples, float(pos_ppm), top_k=int(top_k_samples))
                            if not cand_idx:
                                continue

                            old_mu = float(getattr(n, 'mu', 0.0))
                            old_pi = float(getattr(n, 'pi', 0.0))
                            old_z = getattr(n, 'z_vec', None)
                            old_peak = peak_cache.get(int(n.global_id), None)

                            mu_samples_cpu = mu_samples.detach().cpu().numpy()
                            pi_samples_cpu = pi_samples.detach().cpu().numpy()

                            for sidx in cand_idx:
                                try:
                                    z_cand = z_samples[int(sidx)].detach().clone().to(device)
                                    mu_lib = float(mu_samples_cpu[int(sidx)])
                                    pi_lib = float(pi_samples_cpu[int(sidx)])
                                except Exception:
                                    continue

                                decoded = self._decode_mu_pi_from_z(int(n.su_type), z_cand, g_embed)
                                if decoded is not None:
                                    mu_new, pi_new = decoded
                                else:
                                    mu_new, pi_new = mu_lib, max(1e-6, pi_lib)

                                pk_new = self._peak(mu_new, pi_new, ppm_axis, hwhm=float(hwhm))
                                recon_try = recon
                                if old_peak is not None:
                                    recon_try = recon_try - old_peak
                                recon_try = recon_try + pk_new

                                diff_try = S_t - recon_try
                                loss_try = self._segment_abs(diff_try, region_mask)

                                if float(loss_try) < float(loss_before) - 1e-9:
                                    n.z_vec = z_cand
                                    n.mu = float(mu_new)
                                    n.pi = float(pi_new)
                                    try:
                                        if getattr(n, 'z_history', None) is not None and isinstance(n.z_history, list):
                                            n.z_history.append(z_cand.detach().clone())
                                    except Exception:
                                        pass

                                    peak_cache[int(n.global_id)] = pk_new
                                    recon = recon_try
                                    moves.append({
                                        'region_lo': float(region_lo),
                                        'region_hi': float(region_hi),
                                        'neg_ppm': float(neg_ppm),
                                        'pos_ppm': float(pos_ppm),
                                        'pos_val': float(pos_val),
                                        'global_id': int(n.global_id),
                                        'su_type': int(n.su_type),
                                        'old_mu': float(old_mu),
                                        'new_mu': float(mu_new),
                                        'sample_idx': int(sidx),
                                        'loss_before': float(loss_before),
                                        'loss_after': float(loss_try),
                                    })
                                    accepted = True
                                    break

                            if accepted:
                                break

                        if accepted:
                            break

                    if accepted:
                        break

                if not accepted:
                    break

            if not bool(enable_approx_hop2_template_adjust):
                continue

            max_tpl_iters = int(max(1, int(approx_hop2_max_iters) if approx_hop2_max_iters is not None else int(max_iters)))
            max_dn = int(max(1, int(approx_hop2_max_diff_nodes)))
            max_tpl_k = int(max(1, int(approx_hop2_top_k_templates)))

            for _ in range(max_tpl_iters):
                diff = S_t - recon
                region_mask = self._segment_mask(ppm_axis, float(region_lo), float(region_hi))
                loss_before = self._segment_abs(diff, region_mask)
                neg_ppm_candidates = self._top_negative_ppms(
                    diff=diff,
                    ppm_axis=ppm_axis,
                    mask=region_mask,
                    top_k=int(neg_top_k_peaks),
                    min_sep_ppm=float(neg_peak_min_sep_ppm),
                )
                if not neg_ppm_candidates:
                    break

                accepted = False
                for neg_ppm in neg_ppm_candidates:
                    base_lo = max(float(region_lo), float(neg_ppm) - float(pos_search_window))
                    base_hi = min(float(region_hi), float(neg_ppm) + float(pos_search_window))
                    if base_hi <= base_lo:
                        continue

                    for su_type in pri_su:
                        cand_nodes = [
                            n for n in nodes
                            if int(n.su_type) == int(su_type)
                            and abs(float(getattr(n, 'mu', 0.0)) - float(neg_ppm)) <= float(neg_assign_window)
                            and self._is_h2_approx_node(n)
                        ]
                        cand_nodes.sort(key=lambda n: int(n.global_id))

                        for n in cand_nodes:
                            tpl_key_old = getattr(n, 'template_key', None)
                            if not (isinstance(tpl_key_old, tuple) and len(tpl_key_old) == 3):
                                continue

                            try:
                                center_su = int(n.su_type)
                                hop1_obs = self._multiset_from_counter(getattr(n, 'hop1_su', Counter()) or Counter())
                                hop2_obs = self._multiset_from_counter(getattr(n, 'hop2_su', Counter()) or Counter())
                                tpl_h2_old = tuple(tpl_key_old[2])
                                diff_old = self._multiset_diff_nodes(tuple(hop2_obs), tuple(tpl_h2_old))
                            except Exception:
                                continue

                            center_index = (lib or {}).get('center_index', {}) if isinstance(lib, dict) else {}
                            cand_keys_raw = center_index.get(int(center_su), []) if isinstance(center_index, dict) else []

                            candidates: List[Tuple[int, int, Tuple]] = []
                            for k in cand_keys_raw:
                                kt = tuple(k) if not isinstance(k, tuple) else k
                                if not (isinstance(kt, tuple) and len(kt) == 3):
                                    continue
                                if kt == tpl_key_old:
                                    continue
                                try:
                                    h1_t = tuple(kt[1])
                                    h2_t = tuple(kt[2])
                                except Exception:
                                    continue
                                if tuple(h1_t) != tuple(hop1_obs):
                                    continue
                                dn = self._multiset_diff_nodes(tuple(hop2_obs), tuple(h2_t))
                                if dn <= 0 or dn > max_dn:
                                    continue
                                tpl = templates.get(kt)
                                if not isinstance(tpl, dict):
                                    continue
                                sc = int(tpl.get('sample_count', 0))
                                if sc <= 0:
                                    continue
                                candidates.append((int(dn), -int(sc), kt))

                            if not candidates:
                                continue

                            candidates.sort()
                            if int(diff_old) > 0:
                                candidates = [c for c in candidates if int(c[0]) <= max(int(diff_old), 1)] + [c for c in candidates if int(c[0]) > max(int(diff_old), 1)]
                            candidates = candidates[:max_tpl_k]

                            old_mu = float(getattr(n, 'mu', 0.0))
                            old_pi = float(getattr(n, 'pi', 0.0))
                            old_z = getattr(n, 'z_vec', None)
                            old_peak = peak_cache.get(int(n.global_id), None)

                            for dn, _neg_sc, kt in candidates:
                                tpl = templates.get(kt)
                                if not isinstance(tpl, dict):
                                    continue
                                samples = tpl.get('samples', {})
                                if not isinstance(samples, dict):
                                    continue
                                z_samples = samples.get('z', None)
                                mu_samples = samples.get('mu', None)
                                pi_samples = samples.get('pi', None)
                                if z_samples is None or mu_samples is None or pi_samples is None:
                                    continue
                                if int(mu_samples.numel()) <= 0:
                                    continue

                                mu_min = float(tpl.get('mu_min', 0.0))
                                mu_max = float(tpl.get('mu_max', 0.0))
                                lo = max(float(base_lo), float(mu_min))
                                hi = min(float(base_hi), float(mu_max))
                                if hi <= lo:
                                    continue

                                best_pos = self._best_positive_in_range(diff, ppm_axis, lo, hi)
                                if best_pos is None:
                                    continue
                                pos_ppm, pos_val = best_pos

                                cand_idx = self._mu_sorted_candidates(mu_samples, float(pos_ppm), top_k=int(top_k_samples))
                                if not cand_idx:
                                    continue

                                mu_samples_cpu = mu_samples.detach().cpu().numpy()
                                pi_samples_cpu = pi_samples.detach().cpu().numpy()

                                for sidx in cand_idx:
                                    try:
                                        z_cand = z_samples[int(sidx)].detach().clone().to(device)
                                        mu_lib = float(mu_samples_cpu[int(sidx)])
                                        pi_lib = float(pi_samples_cpu[int(sidx)])
                                    except Exception:
                                        continue

                                    decoded = self._decode_mu_pi_from_z(int(n.su_type), z_cand, g_embed)
                                    if decoded is not None:
                                        mu_new, pi_new = decoded
                                    else:
                                        mu_new, pi_new = mu_lib, max(1e-6, pi_lib)

                                    pk_new = self._peak(mu_new, pi_new, ppm_axis, hwhm=float(hwhm))
                                    recon_try = recon
                                    if old_peak is not None:
                                        recon_try = recon_try - old_peak
                                    recon_try = recon_try + pk_new

                                    diff_try = S_t - recon_try
                                    loss_try = self._segment_abs(diff_try, region_mask)

                                    if float(loss_try) < float(loss_before) - 1e-9:
                                        n.template_key = kt
                                        n.z_vec = z_cand
                                        n.mu = float(mu_new)
                                        n.pi = float(pi_new)
                                        try:
                                            if getattr(n, 'z_history', None) is not None and isinstance(n.z_history, list):
                                                n.z_history.append(z_cand.detach().clone())
                                        except Exception:
                                            pass

                                        peak_cache[int(n.global_id)] = pk_new
                                        recon = recon_try
                                        moves.append({
                                            'move_type': 'approx_hop2_template',
                                            'region_lo': float(region_lo),
                                            'region_hi': float(region_hi),
                                            'neg_ppm': float(neg_ppm),
                                            'pos_ppm': float(pos_ppm),
                                            'pos_val': float(pos_val),
                                            'global_id': int(n.global_id),
                                            'su_type': int(n.su_type),
                                            'old_mu': float(old_mu),
                                            'new_mu': float(mu_new),
                                            'sample_idx': int(sidx),
                                            'loss_before': float(loss_before),
                                            'loss_after': float(loss_try),
                                            'old_template_key': str(tpl_key_old),
                                            'new_template_key': str(kt),
                                            'hop2_diff_nodes': int(dn),
                                        })
                                        accepted = True
                                        break

                                if accepted:
                                    break

                            if accepted:
                                break

                            if tpl_key_old is not None:
                                n.template_key = tpl_key_old
                            if old_z is not None:
                                n.z_vec = old_z
                            n.mu = float(old_mu)
                            n.pi = float(old_pi)
                            if old_peak is not None:
                                peak_cache[int(n.global_id)] = old_peak

                        if accepted:
                            break

                    if accepted:
                        break

                if not accepted:
                    break

        eval_info = evaluate_spectrum_reconstruction(
            S_t,
            recon,
            ppm_axis=ppm_axis,
            fit_scale=True,
            nonnegative_alpha=True,
        )
        S_fit = eval_info['S_fit']
        diff = eval_info['S_target'] - S_fit
        r2 = float(eval_info.get('r2', 0.0))
        alpha = float(eval_info.get('alpha', 1.0))
        
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            df_spec = pd.DataFrame({
                'ppm': ppm_axis.detach().cpu().numpy(),
                'target': S_t.detach().cpu().numpy(),
                'reconstructed_raw': recon.detach().cpu().numpy(),
                'reconstructed': S_fit.detach().cpu().numpy(),
                'difference': diff.detach().cpu().numpy(),
                'alpha': np.full(int(ppm_axis.numel()), float(alpha), dtype=np.float64),
            })
            df_spec.to_csv(str(out / 'layer3_spectrum.csv'), index=False)

        print(f"  R²={r2:.4f}, α={alpha:.4f}")
        print("Layer3 完成\n")
        return nodes, float(r2)
