import torch
import pandas as pd
import copy
import hashlib
import numpy as np
import json
import re
import os
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Sequence

# 导入项目内部模块
from ..shared.coarse_graph import (
    E_SU, SU_PPM_RANGES, NUM_SU_TYPES, PPM_AXIS
)
from ..forward.s2n.s2n_model import S2NConfig, S2NModel
from ..forward.g2s.g2s_model import G2SConfig, NMR_VAE
# 导入拆分模块
from ..shared.inverse_common import (
    _NodeV3, lorentzian_spectrum,
    multiset_from_counter,
    SU_DEFS,
    visualize_su_distribution, evaluate_spectrum_reconstruction, resample_spectrum_to_ppm_axis,
    normalize_spectrum_to_carbon_count,
    SPECIAL_DEGREE_PRIORS,
    get_effective_hist_element_vector,
    get_effective_nodes_element_vector,
    get_node_degree_hint,
    compute_aromatic_area_fraction,
    aromatic_response_scale_from_fraction,
    reconstruction_peak_scale,
)
from .layer0.inverse_layer0 import Layer0Estimator
from .layer1.inverse_layer1 import Layer1Assigner
from .layer2.inverse_layer2 import Layer2Estimator
from .layer3.inverse_layer3 import Layer3Estimator
from .layer4.inverse_layer4 import Layer4Adjuster
from .topology import audit_stage3_topology, classify_topology_units
from .topology.assembly.state.round_context import normalize_stage3_product
from .topology.fixed_routes import stage3_fixed_route_preflight
from .topology.assembly.projection import preflight_fragment_view_projection
from .topology.stage3.flex_audit import _compute_flex_audit
# ============================================================================
# 主推理类
# ============================================================================

class InversePipelineV3:
    """
    改进的逆向推理主类
    
    核心改进：
    1. 软约束系统：约束按优先级分级，允许临时违反低优先级约束
    2. 合成模板生成：库缺失时动态生成化学合理的模板
    3. 增量优化：保留历史信息，避免全盘推翻
    4. 多目标评估：不仅看R²，还考虑分段匹配、尖峰抑制、元素平衡
    """
    
    def __init__(self, 
                 s2n_model: S2NModel,
                 vae_model: NMR_VAE,
                 templates: Optional[Any],
                 device: str = 'cuda'):
        """
        初始化推理管道
        
        Args:
            s2n_model: Spectrum-to-Node模型
            vae_model: NMR VAE模型（用于解码mu/pi）
            templates: 模板库路径或包含'_lib_path'的配置对象
            device: 计算设备
        """
        self.s2n = s2n_model.to(device).eval()
        self.vae = vae_model.to(device).eval()
        self.device = device
        # Chosen once from the target spectrum and applied only to all
        # structure-derived reconstructed spectra.  The target spectrum and
        # topology/resource rules remain untouched.
        self.aromatic_area_fraction: float = 0.0
        self.aromatic_response_scale: float = 1.0
        self.default_template_lib_path: Optional[str] = None
        if isinstance(templates, (str, Path)):
            self.default_template_lib_path = str(Path(templates))
        elif isinstance(templates, dict):
            try:
                lib_path = templates.get('_lib_path')
                if isinstance(lib_path, (str, Path)):
                    self.default_template_lib_path = str(Path(lib_path))
            except Exception:
                pass
        
        # 加载常量到设备
        self.E_SU = E_SU.to(device)
        self.SU_PPM_RANGES = SU_PPM_RANGES  # 字典，不需要.to(device)
        
        # 初始化分层组件（委托模式）
        self.layer0_estimator = Layer0Estimator(
            s2n_model=self.s2n,
            E_SU_tensor=self.E_SU,
            device=device
        )
        self.layer1_assigner = Layer1Assigner(
            device=device,
            E_SU_tensor=self.E_SU,
            layer0_estimator=self.layer0_estimator,  # 传递Layer0实例
        )
        self.layer2_estimator = Layer2Estimator(
            device=torch.device(device),
        )

        self.layer3_estimator = Layer3Estimator(
            device=torch.device(device),
            vae_model=self.vae,
            lib_path=self.default_template_lib_path,
        )
        if self.default_template_lib_path:
            self.layer2_estimator.lib_path = str(self.default_template_lib_path)

        self.layer4_adjuster = Layer4Adjuster(
             device=torch.device(device),
             layer0_estimator=self.layer0_estimator,
         )
        # The final topology report must describe the last committed Stage3
        # transaction, even after Layer3 updates latent/template values.
        self._final_stage3_completion_diagnostics: Dict[str, Any] = {}
        self._final_stage3_terminal_diagnostics: Dict[str, Any] = {}
        self._final_stage3_snapshot_node_ids: List[int] = []
        self._final_stage3_snapshot_authoritative: bool = False

    def _set_aromatic_reconstruction_scale(self, S_target: torch.Tensor) -> None:
        """Configure the shared aromatic peak response for this inference run."""
        fraction = compute_aromatic_area_fraction(S_target, ppm_axis=PPM_AXIS)
        scale = aromatic_response_scale_from_fraction(fraction)
        self.aromatic_area_fraction = float(fraction)
        self.aromatic_response_scale = float(scale)
        self.layer1_assigner.aromatic_response_scale = float(scale)
        hop1_adjuster = getattr(self.layer1_assigner, '_hop1_adjuster', None)
        if hop1_adjuster is not None:
            hop1_adjuster.aromatic_response_scale = float(scale)
        self.layer2_estimator.aromatic_response_scale = float(scale)
        self.layer3_estimator.aromatic_response_scale = float(scale)

    @staticmethod
    def _merge_stage_config(defaults: Dict[str, Any],
                            override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = dict(defaults)
        if isinstance(override, dict):
            cfg.update(override)
        return cfg

    def _set_fixed_partition_meta(self, meta: Optional[Dict[str, Any]]) -> None:
        """Keep Layer4 and Layer0 on the same fixed-anchor metadata snapshot."""
        fixed_meta = copy.deepcopy(meta or {})
        try:
            self.layer4_adjuster.fixed_partition_meta = copy.deepcopy(fixed_meta)
        except Exception:
            pass
        try:
            self.layer0_estimator.fixed_partition_meta = copy.deepcopy(fixed_meta)
            self.layer0_estimator.special_degree_meta = copy.deepcopy(fixed_meta.get('special_degree_meta', {}) or {})
            self.layer0_estimator.special_partition_meta = copy.deepcopy(fixed_meta.get('special_partition_meta', {}) or {})
        except Exception:
            pass

    def _sync_fixed_partition_meta_from_nodes(self, nodes: List[_NodeV3]) -> None:
        """Derive special-degree and fixed-anchor mode metadata from a real graph."""
        fixed_meta = copy.deepcopy(getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {})
        special_degree_meta: Dict[int, Dict[int, int]] = {
            int(su): {int(deg): 0 for deg in dict(priors).keys()}
            for su, priors in SPECIAL_DEGREE_PRIORS.items()
        }
        anchor_mode_meta: Dict[int, Dict[str, Dict[int, int]]] = {
            19: {
                'ether_single': {1: 0, 2: 0, 3: 0},
                'ether_double': {2: 0, 3: 0},
                'thio_single': {1: 0, 2: 0, 3: 0},
                'thio_double': {2: 0, 3: 0},
            },
            20: {
                'single': {1: 0, 2: 0, 3: 0},
                'double': {2: 0, 3: 0},
            },
            21: {
                'single': {2: 0, 3: 0},
                'double': {2: 0, 3: 0},
            },
        }

        for node in list(nodes or []):
            su_i = int(getattr(node, 'su_type', -1))
            if int(su_i) not in SPECIAL_DEGREE_PRIORS:
                continue
            degree = get_node_degree_hint(node)
            if degree is None or int(degree) not in special_degree_meta[int(su_i)]:
                continue
            deg_i = int(degree)
            special_degree_meta[int(su_i)][int(deg_i)] += 1
            fixed_cnt = 0
            try:
                fixed_cnt = max(0, int(getattr(node, 'target_fixed_anchor_count', 0) or 0))
            except Exception:
                fixed_cnt = 0
            if int(su_i) == 19:
                part = str(getattr(node, 'special_anchor_partition', None) or '')
                if str(part) == 'thio':
                    if int(fixed_cnt) >= 2 and int(deg_i) in {2, 3}:
                        anchor_mode_meta[19]['thio_double'][int(deg_i)] += 1
                    else:
                        anchor_mode_meta[19]['thio_single'][int(deg_i)] += 1
                elif str(part) == 'ether':
                    if int(fixed_cnt) >= 2 and int(deg_i) in {2, 3}:
                        anchor_mode_meta[19]['ether_double'][int(deg_i)] += 1
                    else:
                        anchor_mode_meta[19]['ether_single'][int(deg_i)] += 1
            elif int(su_i) == 20:
                if int(fixed_cnt) >= 2 and int(deg_i) in {2, 3}:
                    anchor_mode_meta[20]['double'][int(deg_i)] += 1
                else:
                    anchor_mode_meta[20]['single'][int(deg_i)] += 1
            elif int(su_i) == 21:
                if int(fixed_cnt) >= 2 and int(deg_i) in {2, 3}:
                    anchor_mode_meta[21]['double'][int(deg_i)] += 1
                else:
                    anchor_mode_meta[21]['single'][int(deg_i)] += 1

        ether_by_degree = {
            int(deg): (
                int(anchor_mode_meta[19]['ether_single'].get(int(deg), 0)) +
                int(anchor_mode_meta[19]['ether_double'].get(int(deg), 0))
            )
            for deg in [1, 2, 3]
        }
        thio_by_degree = {
            int(deg): (
                int(anchor_mode_meta[19]['thio_single'].get(int(deg), 0)) +
                int(anchor_mode_meta[19]['thio_double'].get(int(deg), 0))
            )
            for deg in [1, 2, 3]
        }
        fixed_meta['special_degree_meta'] = special_degree_meta
        fixed_meta['special_anchor_mode_meta'] = anchor_mode_meta
        fixed_meta['special_partition_meta'] = dict(fixed_meta.get('special_partition_meta', {}) or {})
        fixed_meta['special_partition_meta'][19] = {
            'ether': dict(ether_by_degree),
            'thio': dict(thio_by_degree),
        }
        fixed_meta['o_base_19'] = int(sum(int(v) for v in ether_by_degree.values()))
        fixed_meta['s_reserved_19'] = int(sum(int(v) for v in thio_by_degree.values()))
        fixed_meta['n19_total'] = int(fixed_meta['o_base_19'] + fixed_meta['s_reserved_19'])
        self._set_fixed_partition_meta(fixed_meta)

    def _build_layer_runtime_configs(
        self,
        layer1_config: Optional[Dict[str, Any]] = None,
        layer3_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        layer1_defaults = {
            'neg_peak_threshold': -0.8,
            'pos_peak_threshold': 0.8,
            'max_candidate_trials_per_region': 40,
            'max_moves_per_region': 6,
            'region_adjustment_cycles': 2,
            'layer1_multistart_count': 6,
            'layer1_multistart_refine_count': 2,
            'layer1_multistart_branch_limit': 12,
            'verbose': False,
        }
        layer3_defaults = {
            'max_iters': 40,
            'pos_search_window': 15.0,
            'neg_assign_window': 2.0,
            'top_k_samples': 10,
            'neg_top_k_peaks': 15,
            'neg_peak_min_sep_ppm': 1.0,
            'layer3_verbose': True,
        }

        def _merge_strict(
            defaults: Dict[str, Any],
            user_config: Optional[Dict[str, Any]],
            label: str,
        ) -> Dict[str, Any]:
            config = dict(user_config or {})
            unsupported = sorted(
                str(key) for key in config if str(key) not in defaults
            )
            if unsupported:
                raise ValueError(f'{label} contains unsupported keys: {unsupported}')
            merged = dict(defaults)
            merged.update(config)
            return merged

        layer1_merged = _merge_strict(
            layer1_defaults,
            layer1_config,
            'layer1_config',
        )
        for key in (
            'region_adjustment_cycles',
            'layer1_multistart_count',
            'layer1_multistart_refine_count',
        ):
            if int(layer1_merged[key]) < 1:
                raise ValueError(f'{key} must be >= 1')
        if int(layer1_merged['layer1_multistart_branch_limit']) < 0:
            raise ValueError('layer1_multistart_branch_limit must be >= 0')
        for key in (
            'max_candidate_trials_per_region',
            'max_moves_per_region',
        ):
            if int(layer1_merged[key]) < 0:
                raise ValueError(f'{key} must be >= 0')

        return (
            layer1_merged,
            _merge_strict(layer3_defaults, layer3_config, 'layer3_config'),
        )

    def get_default_runtime_configs(self) -> Dict[str, Dict[str, Any]]:
        """Return the supported inverse Layer1 and Layer3 runtime settings."""
        layer1_config, layer3_config = self._build_layer_runtime_configs()
        return {
            'layer1_config': layer1_config,
            'layer3_config': layer3_config,
        }

    @staticmethod
    def _stage3_snapshot_json_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, set):
            return [
                InversePipelineV3._stage3_snapshot_json_value(item)
                for item in sorted(value, key=str)
            ]
        if isinstance(value, dict):
            return {
                str(key): InversePipelineV3._stage3_snapshot_json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                InversePipelineV3._stage3_snapshot_json_value(item)
                for item in value
            ]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, 'to_dict'):
            try:
                return InversePipelineV3._stage3_snapshot_json_value(
                    value.to_dict()
                )
            except Exception:
                pass
        return str(value)

    @classmethod
    def _write_stage3_snapshot_json(cls, path: Path, value: Any) -> None:
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(
                cls._stage3_snapshot_json_value(value),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

    @staticmethod
    def _stage3_snapshot_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _restore_numeric_json_keys(value: Any) -> Any:
        if isinstance(value, list):
            return [
                InversePipelineV3._restore_numeric_json_keys(item)
                for item in value
            ]
        if not isinstance(value, dict):
            return value
        restored: Dict[Any, Any] = {}
        for raw_key, item in value.items():
            key: Any = raw_key
            if isinstance(raw_key, str) and raw_key.lstrip('-').isdigit():
                key = int(raw_key)
            restored[key] = InversePipelineV3._restore_numeric_json_keys(item)
        return restored

    @staticmethod
    def _load_stage3_snapshot_torch(path: Path, *, map_location: Any) -> Any:
        try:
            return torch.load(
                path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            return torch.load(path, map_location=map_location)

    def _archive_stage3_snapshot_path(
        self,
        snapshot_path: Path,
        *,
        run_id: str,
    ) -> Optional[Path]:
        if not snapshot_path.exists():
            return None
        archive_dir = snapshot_path.parent / 'archive'
        archive_dir.mkdir(parents=True, exist_ok=True)
        target = archive_dir / f'{snapshot_path.name}-{run_id}'
        suffix = 1
        while target.exists():
            target = archive_dir / f'{snapshot_path.name}-{run_id}-{suffix}'
            suffix += 1
        shutil.move(str(snapshot_path), str(target))
        return target

    def _save_stage3_input_snapshot(
        self,
        *,
        output_dir: str,
        nodes: Sequence[_NodeV3],
        H: torch.Tensor,
        S_target: torch.Tensor,
        E_target: torch.Tensor,
        current_diff: Dict[str, Any],
        fixed_partition_meta: Dict[str, Any],
        layer1_runtime_cfg: Dict[str, Any],
        stage3_resource_cfg: Dict[str, Any],
        eval_lib_path: str,
        outer_idx: int,
        run_id: str,
    ) -> Path:
        baseline = dict(current_diff or {})
        if (
            str(baseline.get('source_stage', '') or '') != 'layer2'
            or str(baseline.get('spectral_contract', '') or '')
            != 'template_center_mu'
        ):
            raise ValueError('stage3_snapshot_requires_saved_layer2_baseline')

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        snapshot_dir = output_root / 'stage3_input_snapshot'
        self._archive_stage3_snapshot_path(snapshot_dir, run_id=run_id)
        temp_dir = output_root / (
            f'.stage3_input_snapshot.tmp-{int(os.getpid())}-{time.time_ns()}'
        )
        temp_dir.mkdir(parents=False, exist_ok=False)
        try:
            snapshot_nodes = copy.deepcopy(list(nodes or []))
            torch.save(snapshot_nodes, temp_dir / 'nodes.pt')
            torch.save(H.detach().cpu().clone(), temp_dir / 'H.pt')
            torch.save(
                S_target.detach().cpu().clone(), temp_dir / 'S_target.pt'
            )
            torch.save(
                E_target.detach().cpu().clone(), temp_dir / 'E_target.pt'
            )
            self._write_stage3_snapshot_json(
                temp_dir / 'current_diff.json', baseline
            )
            self._write_stage3_snapshot_json(
                temp_dir / 'fixed_partition_meta.json', fixed_partition_meta
            )
            self._write_stage3_snapshot_json(
                temp_dir / 'layer1_runtime_config.json', layer1_runtime_cfg
            )
            self._write_stage3_snapshot_json(
                temp_dir / 'stage3_resource_config.json', stage3_resource_cfg
            )

            node_rows: List[Dict[str, Any]] = []
            node_hops: Dict[str, Dict[str, Any]] = {}
            for node in sorted(
                snapshot_nodes,
                key=lambda item: int(getattr(item, 'global_id', -1)),
            ):
                node_id = int(getattr(node, 'global_id', -1))
                hop1_su = {
                    str(key): int(value)
                    for key, value in dict(
                        getattr(node, 'hop1_su', {}) or {}
                    ).items()
                }
                hop2_su = {
                    str(key): int(value)
                    for key, value in dict(
                        getattr(node, 'hop2_su', {}) or {}
                    ).items()
                }
                hop_record = {
                    'hop1_ids': sorted(
                        int(value)
                        for value in list(getattr(node, 'hop1_ids', []) or [])
                    ),
                    'hop2_ids': sorted(
                        int(value)
                        for value in list(getattr(node, 'hop2_ids', []) or [])
                    ),
                    'fixed_hop1_ids': sorted(
                        int(value)
                        for value in list(
                            getattr(node, 'fixed_hop1_ids', set()) or set()
                        )
                    ),
                    'hop1_su': hop1_su,
                    'hop2_su': hop2_su,
                }
                node_hops[str(node_id)] = hop_record
                z_vec = getattr(node, 'z_vec', None)
                node_rows.append({
                    'global_id': node_id,
                    'su_type': int(getattr(node, 'su_type', -1)),
                    'mu': float(getattr(node, 'mu', 0.0) or 0.0),
                    'pi': float(getattr(node, 'pi', 0.0) or 0.0),
                    'template_key': str(getattr(node, 'template_key', None)),
                    'target_hop1_degree': getattr(
                        node, 'target_hop1_degree', None
                    ),
                    'target_fixed_anchor_count': getattr(
                        node, 'target_fixed_anchor_count', None
                    ),
                    'special_anchor_partition': str(
                        getattr(node, 'special_anchor_partition', '') or ''
                    ),
                    'special_anchor_mode': str(
                        getattr(node, 'special_anchor_mode', '') or ''
                    ),
                    'hop1_ids': json.dumps(hop_record['hop1_ids']),
                    'hop2_ids': json.dumps(hop_record['hop2_ids']),
                    'fixed_hop1_ids': json.dumps(
                        hop_record['fixed_hop1_ids']
                    ),
                    'hop1_su': json.dumps(hop1_su, sort_keys=True),
                    'hop2_su': json.dumps(hop2_su, sort_keys=True),
                    'z_vec': json.dumps(
                        z_vec.detach().cpu().flatten().tolist()
                        if isinstance(z_vec, torch.Tensor) else []
                    ),
                })
            pd.DataFrame(node_rows).to_csv(
                temp_dir / 'nodes_by_global_id.csv', index=False
            )
            self._write_stage3_snapshot_json(
                temp_dir / 'node_hops.json', node_hops
            )

            data_files = sorted(
                path.name
                for path in temp_dir.iterdir()
                if path.is_file()
            )
            manifest = {
                'format': 'graphnmr_stage3_input_snapshot',
                'version': 1,
                'created_at_unix': float(time.time()),
                'run_id': str(run_id),
                'outer_idx': int(outer_idx),
                'node_count': int(len(snapshot_nodes)),
                'source_stage': 'layer2',
                'spectral_contract': 'template_center_mu',
                'eval_lib_path': str(eval_lib_path),
                'files': data_files,
                'sha256': {
                    name: self._stage3_snapshot_sha256(temp_dir / name)
                    for name in data_files
                },
            }
            self._write_stage3_snapshot_json(
                temp_dir / 'snapshot_manifest.json', manifest
            )
            temp_dir.rename(snapshot_dir)
        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise
        return snapshot_dir

    def replay_stage3_from_snapshot(
        self,
        snapshot_dir: str | Path,
        output_dir: Optional[str] = None,
        stage_configs: Optional[Dict[str, Any]] = None,
        outer_idx: int = 0,
    ) -> Tuple[List[_NodeV3], torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        """Run Stage3 directly from a saved pre-Stage3 Layer2 boundary."""
        snapshot_path = Path(snapshot_dir)
        manifest_path = snapshot_path / 'snapshot_manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f'missing Stage3 snapshot manifest: {manifest_path}'
            )
        with open(manifest_path, 'r', encoding='utf-8') as handle:
            manifest = dict(json.load(handle) or {})
        if str(manifest.get('format', '')) != 'graphnmr_stage3_input_snapshot':
            raise ValueError('invalid_stage3_snapshot_format')
        for name, expected in dict(manifest.get('sha256', {}) or {}).items():
            file_path = snapshot_path / str(name)
            if not file_path.is_file():
                raise FileNotFoundError(
                    f'missing Stage3 snapshot file: {file_path}'
                )
            if self._stage3_snapshot_sha256(file_path) != str(expected):
                raise ValueError(f'stage3_snapshot_checksum_mismatch:{name}')

        nodes = self._load_stage3_snapshot_torch(
            snapshot_path / 'nodes.pt', map_location=self.device
        )
        H = self._load_stage3_snapshot_torch(
            snapshot_path / 'H.pt', map_location=self.device
        ).to(self.device)
        S_target = self._load_stage3_snapshot_torch(
            snapshot_path / 'S_target.pt', map_location=self.device
        ).to(self.device)
        E_target = self._load_stage3_snapshot_torch(
            snapshot_path / 'E_target.pt', map_location=self.device
        ).to(self.device)
        with open(snapshot_path / 'current_diff.json', 'r', encoding='utf-8') as handle:
            current_diff = dict(json.load(handle) or {})
        # Recompute the Layer2 baseline under the current reconstruction
        # response.  Older snapshots may have been written before aromatic
        # response metadata existed; the target remains unchanged.
        self._set_aromatic_reconstruction_scale(S_target)
        stored_diff_contract = {
            key: current_diff[key]
            for key in ('source_stage', 'spectral_contract', 'mode')
            if key in current_diff
        }
        current_diff = self._compute_difference_spectrum_from_nodes_mu(
            nodes,
            S_target,
        )
        current_diff.update(stored_diff_contract)
        with open(
            snapshot_path / 'fixed_partition_meta.json', 'r', encoding='utf-8'
        ) as handle:
            fixed_partition_meta = self._restore_numeric_json_keys(
                json.load(handle)
            )
        with open(
            snapshot_path / 'layer1_runtime_config.json', 'r', encoding='utf-8'
        ) as handle:
            layer1_runtime_cfg = dict(json.load(handle) or {})
        with open(
            snapshot_path / 'stage3_resource_config.json', 'r', encoding='utf-8'
        ) as handle:
            resource_cfg = dict(json.load(handle) or {})

        override = dict(stage_configs or {})
        if override:
            if 'block_c_resource' in override:
                unknown_sections = set(override) - {'block_c_resource'}
                if unknown_sections:
                    raise ValueError(
                        'replay only accepts block_c_resource overrides: '
                        + ', '.join(sorted(str(key) for key in unknown_sections))
                    )
                resource_cfg.update(
                    dict(override.get('block_c_resource', {}) or {})
                )
            else:
                resource_cfg.update(override)
        if not bool(resource_cfg.pop('enabled', True)):
            raise ValueError('replay_stage3_requires_block_c_resource_enabled')

        supported_resource_keys = {
            'flex_max_primary_adjustments',
            'external_max_primary_adjustments',
            'internal_max_primary_adjustments',
            'max_h_compensation_ops_per_adjustment',
            'residual_flex_preferred_23',
            'residual_flex_min_23',
            'residual_side_preferred_23',
            'residual_side_min_23',
            'h_tolerance',
            'enable_post_external_spectrum',
            'spectrum_active_through_cycle',
            'spectrum_candidate_limit',
            'spectrum_max_candidate_trials',
            'spectrum_max_primary_adjustments',
            'spectrum_min_r2_gain',
            'verbose',
        }
        unknown_keys = set(resource_cfg) - supported_resource_keys
        if unknown_keys:
            raise ValueError(
                'block_c_resource contains unknown replay keys: '
                + ', '.join(sorted(str(key) for key in unknown_keys))
            )
        if len(list(nodes or [])) != int(manifest.get('node_count', -1)):
            raise ValueError('stage3_snapshot_node_count_mismatch')
        if (
            str(current_diff.get('source_stage', '') or '') != 'layer2'
            or str(current_diff.get('spectral_contract', '') or '')
            != 'template_center_mu'
        ):
            raise ValueError('stage3_snapshot_layer2_contract_invalid')

        self._set_fixed_partition_meta(fixed_partition_meta)
        resolved_lib = str(
            manifest.get('eval_lib_path', '')
            or self.default_template_lib_path
            or ''
        )
        if not resolved_lib:
            raise ValueError('stage3_snapshot_missing_template_library')
        self.layer1_assigner._configure_exact_hop1_shift_constraints(resolved_lib)
        hop1_adjuster = getattr(self.layer1_assigner, '_hop1_adjuster', None)
        self.layer4_adjuster.set_spectrum_hop1_template_options(
            getattr(hop1_adjuster, 'shift_options_by_su', {}) or {}
        )
        self.layer4_adjuster._spectrum_search_memory = {}

        replay_output = None
        if output_dir:
            replay_root = Path(output_dir)
            replay_root.mkdir(parents=True, exist_ok=True)
            replay_output = str(replay_root / 'block_c')
        result = self._run_topology_stage3_resource_cycles(
            nodes=copy.deepcopy(list(nodes or [])),
            H=H.detach().clone(),
            current_diff=copy.deepcopy(current_diff),
            S_target=S_target,
            E_target=E_target,
            eval_lib_path=resolved_lib,
            layer1_runtime_cfg=layer1_runtime_cfg,
            outer_idx=int(outer_idx),
            flex_max_primary_adjustments=int(resource_cfg.get('flex_max_primary_adjustments', 10)),
            external_max_primary_adjustments=int(resource_cfg.get('external_max_primary_adjustments', 10)),
            internal_max_primary_adjustments=int(resource_cfg.get('internal_max_primary_adjustments', 30)),
            max_h_compensation_ops_per_adjustment=int(resource_cfg.get('max_h_compensation_ops_per_adjustment', 8)),
            residual_flex_preferred_23=int(resource_cfg.get('residual_flex_preferred_23', 3)),
            residual_flex_min_23=int(resource_cfg.get('residual_flex_min_23', 2)),
            residual_side_preferred_23=int(resource_cfg.get('residual_side_preferred_23', 2)),
            residual_side_min_23=int(resource_cfg.get('residual_side_min_23', 1)),
            h_tolerance=float(resource_cfg.get('h_tolerance', 0.08)),
            enable_post_external_spectrum=bool(resource_cfg.get('enable_post_external_spectrum', True)),
            spectrum_active_through_cycle=int(resource_cfg.get('spectrum_active_through_cycle', 20)),
            spectrum_candidate_limit=int(resource_cfg.get('spectrum_candidate_limit', 8)),
            spectrum_max_candidate_trials=int(resource_cfg.get('spectrum_max_candidate_trials', 4)),
            spectrum_max_primary_adjustments=int(resource_cfg.get('spectrum_max_primary_adjustments', 10)),
            spectrum_min_r2_gain=float(resource_cfg.get('spectrum_min_r2_gain', 0.0005)),
            verbose=bool(resource_cfg.get('verbose', False)),
            output_dir=replay_output,
        )
        replay_nodes, replay_H, replay_diff, replay_meta = result
        if output_dir:
            replay_root = Path(output_dir)
            torch.save(copy.deepcopy(replay_nodes), replay_root / 'nodes.pt')
            torch.save(replay_H.detach().cpu().clone(), replay_root / 'H.pt')
            self._write_stage3_snapshot_json(
                replay_root / 'current_diff.json', replay_diff
            )
            self._write_stage3_snapshot_json(
                replay_root / 'stage3_meta.json', replay_meta
            )
        return replay_nodes, replay_H, replay_diff, replay_meta

    @staticmethod
    def _format_nonzero_histogram(H: torch.Tensor) -> str:
        try:
            vals = H.detach().cpu().long().tolist()
        except Exception:
            vals = [int(x) for x in list(H)]
        parts = [f"SU{idx}:{int(cnt)}" for idx, cnt in enumerate(vals) if int(cnt) > 0]
        return ", ".join(parts) if parts else "(empty)"

    @staticmethod
    def _format_special_degree_distribution(nodes: Optional[List[_NodeV3]]) -> Optional[str]:
        if not nodes:
            return None
        parts: List[str] = []
        for su_type in (19, 20, 21):
            deg_keys = sorted(int(k) for k in dict(SPECIAL_DEGREE_PRIORS.get(int(su_type), {})).keys())
            counts = {int(deg): 0 for deg in deg_keys}
            unknown = 0
            total = 0
            for node in nodes:
                if int(getattr(node, 'su_type', -1)) != int(su_type):
                    continue
                total += 1
                deg = get_node_degree_hint(node)
                if deg is None or int(deg) not in counts:
                    unknown += 1
                    continue
                counts[int(deg)] += 1
            if total <= 0:
                continue
            deg_txt = ", ".join(f"d{int(deg)}:{int(counts[deg])}" for deg in deg_keys)
            if int(unknown) > 0:
                deg_txt = f"{deg_txt}, d?:{int(unknown)}"
            parts.append(f"SU{int(su_type)}[{deg_txt}]")
        return "; ".join(parts) if parts else None

    def _print_histogram_and_elements(self,
                                      H: torch.Tensor,
                                      E_target: Optional[torch.Tensor],
                                      label: str,
                                      nodes: Optional[List[_NodeV3]] = None) -> None:
        try:
            H_cpu = H.detach().cpu().long()
            if nodes:
                E_pred = get_effective_nodes_element_vector(
                    nodes,
                    self.E_SU.detach().cpu(),
                    device=torch.device('cpu'),
                )
            else:
                try:
                    special_degree_meta = self.layer4_adjuster._get_special_degree_meta(H_cpu)
                except Exception:
                    special_degree_meta = None
                E_pred = get_effective_hist_element_vector(
                    H_cpu,
                    special_degree_meta=special_degree_meta,
                    E_SU_tensor=self.E_SU.detach().cpu(),
                    device=torch.device('cpu'),
                )
            if E_target is not None:
                E_t = E_target.detach().cpu().flatten().float()
                print(f"[{label}] H(nonzero) = {self._format_nonzero_histogram(H_cpu)}")
                degree_dist = self._format_special_degree_distribution(nodes)
                if degree_dist:
                    print(f"[{label}] SU19/20/21 degree_dist = {degree_dist}")
                print(
                    f"[{label}] Elements pred = {[float(x) for x in E_pred.tolist()]} "
                    f"target = {[float(x) for x in E_t.tolist()]}"
                )
            else:
                print(f"[{label}] H(nonzero) = {self._format_nonzero_histogram(H_cpu)}")
                degree_dist = self._format_special_degree_distribution(nodes)
                if degree_dist:
                    print(f"[{label}] SU19/20/21 degree_dist = {degree_dist}")
                print(f"[{label}] Elements pred = {[float(x) for x in E_pred.tolist()]}")
        except Exception as e:
            print(f"[{label}] Summary unavailable: {e}")

    def _print_effective_resource_summary(self,
                                          H: torch.Tensor,
                                          label: str) -> None:
        try:
            H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
            out = {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0}
            for idx in range(int(H_cpu.numel())):
                cnt = int(H_cpu[idx].item())
                if cnt <= 0:
                    continue
                if int(idx) in {5, 6, 7, 8, 9, 11}:
                    out['11'] += int(cnt)
                elif int(idx) in {1, 4, 16, 18, 22, 28, 32}:
                    out['22'] += int(cnt)
                elif int(idx) in {14, 24}:
                    out['24'] += int(cnt)
                elif int(idx) == 25:
                    out['25'] += int(cnt)
                elif int(idx) in {0, 2, 3, 15, 17, 19, 20, 21, 23, 27, 29, 31}:
                    out['23'] += int(cnt)
            print(
                f"[{label}] Effective resources = "
                f"11:{int(out['11'])} 22:{int(out['22'])} 23:{int(out['23'])} "
                f"24:{int(out['24'])} 25:{int(out['25'])}"
            )
        except Exception as e:
            print(f"[{label}] Effective resource summary unavailable: {e}")

    def _hist_element_counts_from_H(self, H: torch.Tensor) -> Dict[str, float]:
        labels = ('C', 'H', 'O', 'N', 'S', 'X')
        out = {str(label): 0.0 for label in labels}
        try:
            H_cpu = H.detach().cpu().flatten().long()
            vec = get_effective_hist_element_vector(
                H_cpu,
                E_SU_tensor=self.E_SU.detach().cpu(),
                device=torch.device('cpu'),
            ).detach().cpu().flatten().float()
            out = {
                str(label): float(vec[idx].item())
                for idx, label in enumerate(labels)
                if idx < int(vec.numel())
            }
        except Exception:
            pass
        return out

    def _element_error_summary(self,
                               H: Optional[torch.Tensor],
                               E_target: Optional[torch.Tensor],
                               nodes: Optional[List[_NodeV3]] = None) -> Dict[str, Any]:
        labels = ('C', 'H', 'O', 'N', 'S', 'X')
        pred = {str(label): 0.0 for label in labels}
        try:
            if nodes:
                vec = get_effective_nodes_element_vector(
                    list(nodes or []),
                    self.E_SU.detach().cpu(),
                    device=torch.device('cpu'),
                )
                pred = {
                    str(label): float(vec.detach().cpu().flatten()[idx].item())
                    for idx, label in enumerate(labels)
                    if idx < int(vec.numel())
                }
            elif H is not None:
                H_cpu = H.detach().cpu().long()
                try:
                    special_degree_meta = self.layer4_adjuster._get_special_degree_meta(H_cpu)
                except Exception:
                    special_degree_meta = None
                vec = get_effective_hist_element_vector(
                    H_cpu,
                    special_degree_meta=special_degree_meta,
                    E_SU_tensor=self.E_SU.detach().cpu(),
                    device=torch.device('cpu'),
                )
                pred = {
                    str(label): float(vec.detach().cpu().flatten()[idx].item())
                    for idx, label in enumerate(labels)
                    if idx < int(vec.numel())
                }
        except Exception:
            pred = self._hist_element_counts_from_H(H) if H is not None else pred
        target: Dict[str, float] = {}
        if E_target is not None:
            try:
                t = E_target.detach().cpu().flatten().float()
                target = {
                    str(label): float(t[idx].item())
                    for idx, label in enumerate(labels)
                    if idx < int(t.numel())
                }
            except Exception:
                target = {}
        errors = {
            str(label): float(pred.get(str(label), 0.0) - target.get(str(label), 0.0))
            for label in labels
            if str(label) in pred and str(label) in target
        }
        abs_total = float(sum(abs(float(v)) for v in errors.values()))
        return {
            'pred': pred,
            'target': target,
            'error': errors,
            'abs_total_error': abs_total,
            'h_error': float(errors.get('H', 0.0)),
        }

    def _run_layer12_cycle(
        self,
        H_seed: torch.Tensor,
        S_target: torch.Tensor,
        E_target: torch.Tensor,
        eval_lib_path: Optional[str],
        layer1_runtime_cfg: Dict[str, Any],
        eval_nmr: bool = False,
        eval_output_dir: Optional[str] = None,
        layer2_output_dir: Optional[str] = None,
        identity_seed_nodes: Optional[List[_NodeV3]] = None,
        emit_summary: bool = True,
    ) -> Tuple[List[_NodeV3], torch.Tensor, Dict[str, object], float]:
        if not eval_lib_path:
            raise ValueError('Layer1 requires eval_lib_path for exact shift lookup')
        fixed_meta = copy.deepcopy(
            getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {}
        )
        if fixed_meta:
            self._set_fixed_partition_meta(fixed_meta)

        started_at = time.perf_counter()
        runtime_cfg = dict(layer1_runtime_cfg or {})
        multistart_count = int(runtime_cfg.pop('layer1_multistart_count', 6))
        multistart_refine_count = int(runtime_cfg.pop(
            'layer1_multistart_refine_count', 2
        ))
        multistart_branch_limit = int(runtime_cfg.pop(
            'layer1_multistart_branch_limit', 12
        ))
        verbose = bool(runtime_cfg.get('verbose', False))

        variants, generation_diagnostics = (
            self.layer1_assigner.build_rule_valid_graph_variants(
                H_init=H_seed.detach().clone(),
                E_target=E_target,
                eval_lib_path=str(eval_lib_path),
                identity_seed_nodes=(
                    copy.deepcopy(identity_seed_nodes)
                    if identity_seed_nodes is not None
                    else None
                ),
                candidate_count=int(multistart_count),
                branch_limit=int(multistart_branch_limit),
                verbose=bool(verbose),
            )
        )
        adjuster = self.layer1_assigner._hop1_adjuster
        if adjuster is None:
            raise RuntimeError('Layer1 exact template service was not initialized')

        carbon_su_set = set(
            int(value)
            for value in getattr(
                self.layer1_assigner,
                '_carbon_su_set',
                set(),
            )
        )

        def _unique_hop1_motif_count(candidate_nodes: Sequence[_NodeV3]) -> int:
            motifs = set()
            for node in list(candidate_nodes or []):
                su_type = int(getattr(node, 'su_type', -1))
                if carbon_su_set and su_type not in carbon_su_set:
                    continue
                hop1_su = getattr(node, 'hop1_su', None)
                if hasattr(hop1_su, 'elements'):
                    hop1_tuple = tuple(sorted(
                        int(value) for value in hop1_su.elements()
                    ))
                else:
                    hop1_tuple = tuple(sorted(
                        int(value)
                        for value in list(getattr(node, 'hop1_ids', []) or [])
                    ))
                motifs.add((int(su_type), hop1_tuple))
            return int(len(motifs))

        def _rank_float(value: Any) -> float:
            try:
                result = float(value)
            except Exception:
                return float('-inf')
            return result if np.isfinite(result) else float('-inf')

        candidate_records: List[Dict[str, Any]] = []
        diagnostics_rows: List[Dict[str, Any]] = []
        for variant in list(variants or []):
            row: Dict[str, Any] = {
                'variant_id': str(variant.get('variant_id', '')),
                'branch': copy.deepcopy(variant.get('branch')),
                'graph_signature': str(variant.get('graph_signature', '')),
                'raw_layer1_r2': None,
                'raw_layer2_r2': None,
                'final_layer1_r2': None,
                'final_layer2_r2': None,
                'unique_hop1_motif_count': 0,
                'selected_for_refinement': False,
                'selected': False,
                'status': 'raw_pending',
                'failure_reason': '',
            }
            diagnostics_rows.append(row)
            try:
                raw_layer1_nodes = copy.deepcopy(list(variant.get('nodes', []) or []))
                coverage = self.layer1_assigner.audit_layer1_shift_coverage(
                    raw_layer1_nodes,
                    str(eval_lib_path),
                )
                if not bool(coverage.get('ok', False)):
                    raise RuntimeError(
                        'rule_valid_but_shift_template_missing: '
                        f"missing={list(coverage.get('missing_signatures', []) or [])[:8]}"
                    )
                raw_layer1_snapshot = adjuster.build_eval_snapshot(
                    raw_layer1_nodes,
                    S_target,
                )
                if int(raw_layer1_snapshot.get('matched_cnt', 0)) != int(
                    raw_layer1_snapshot.get('carbon_cnt', 0)
                ):
                    raise RuntimeError(
                        'layer1_multistart_raw_template_coverage_failed: '
                        f"matched={raw_layer1_snapshot.get('matched_cnt', 0)}/"
                        f"{raw_layer1_snapshot.get('carbon_cnt', 0)}"
                    )
                adjuster.apply_assignments_to_nodes(
                    raw_layer1_nodes,
                    raw_layer1_snapshot.get('assignments', {}),
                )
                raw_layer2_nodes = self.layer2_assign(
                    nodes=copy.deepcopy(raw_layer1_nodes),
                    S_target=S_target,
                    E_target=E_target,
                    lib_path=eval_lib_path,
                    output_dir=None,
                    verbose=False,
                    emit_summary=False,
                )
                raw_layer2_diff = self._compute_difference_spectrum_from_nodes_mu(
                    nodes=raw_layer2_nodes,
                    S_target=S_target,
                )
                row.update({
                    'raw_layer1_r2': float(raw_layer1_snapshot.get('r2', 0.0)),
                    'raw_layer2_r2': float(raw_layer2_diff.get('r2', 0.0)),
                    'unique_hop1_motif_count': _unique_hop1_motif_count(
                        raw_layer1_nodes
                    ),
                    'status': 'raw_scored',
                })
                candidate_records.append({
                    'variant': variant,
                    'diagnostics': row,
                    'raw_layer1_nodes': raw_layer1_nodes,
                })
            except Exception as exc:
                row['status'] = 'raw_failed'
                row['failure_reason'] = str(exc)

        if not candidate_records:
            failures = ' | '.join(
                f"{row['variant_id']}: {row['failure_reason']}"
                for row in diagnostics_rows[:8]
            )
            raise RuntimeError(
                'layer1_multistart_no_raw_candidate_succeeded: ' + failures
            )

        candidate_records.sort(key=lambda item: (
            -_rank_float(item['diagnostics']['raw_layer2_r2']),
            -_rank_float(item['diagnostics']['raw_layer1_r2']),
            -int(item['diagnostics']['unique_hop1_motif_count']),
            str(item['diagnostics']['graph_signature']),
        ))
        refine_count = min(
            len(candidate_records),
            max(1, int(multistart_refine_count)),
        )
        refinement_failures: List[str] = []
        refined_records: List[Dict[str, Any]] = []
        for candidate in candidate_records[:refine_count]:
            row = candidate['diagnostics']
            row['selected_for_refinement'] = True
            row['status'] = 'refine_pending'
            try:
                refined_layer1_nodes = self.layer1_assigner.layer1_assign(
                    H_init=H_seed.detach().clone(),
                    S_target=S_target,
                    E_target=E_target,
                    eval_nmr=False,
                    eval_output_dir=None,
                    eval_lib_path=str(eval_lib_path),
                    prebuilt_nodes=copy.deepcopy(candidate['raw_layer1_nodes']),
                    emit_summary=False,
                    **runtime_cfg,
                )
                valid, errors = self.layer1_assigner.validate_graph_consistency(
                    nodes=refined_layer1_nodes,
                    H=H_seed.detach().clone(),
                    E_target=E_target,
                    verbose=False,
                )
                if not valid or errors:
                    raise RuntimeError(
                        'layer1_multistart_refined_graph_invalid: '
                        + ' | '.join(
                            str(error) for error in list(errors or [])[:8]
                        )
                    )
                refined_layer1_snapshot = adjuster.build_eval_snapshot(
                    refined_layer1_nodes,
                    S_target,
                )
                region_adjustment = copy.deepcopy(
                    getattr(
                        self.layer1_assigner,
                        '_last_hop1_adjust_diagnostics',
                        {},
                    ) or {}
                )
                shift_coverage = copy.deepcopy(
                    getattr(
                        self.layer1_assigner,
                        '_last_shift_coverage_audit',
                        {},
                    ) or {}
                )
                refined_layer2_nodes = self.layer2_assign(
                    nodes=copy.deepcopy(refined_layer1_nodes),
                    S_target=S_target,
                    E_target=E_target,
                    lib_path=eval_lib_path,
                    output_dir=None,
                    verbose=False,
                    emit_summary=False,
                )
                refined_layer2_diff = self._compute_difference_spectrum_from_nodes_mu(
                    nodes=refined_layer2_nodes,
                    S_target=S_target,
                )
                row.update({
                    'final_layer1_r2': float(
                        refined_layer1_snapshot.get('r2', 0.0)
                    ),
                    'final_layer2_r2': float(refined_layer2_diff.get('r2', 0.0)),
                    'unique_hop1_motif_count': _unique_hop1_motif_count(
                        refined_layer1_nodes
                    ),
                    'final_graph_signature': (
                        self.layer1_assigner._graph_signature(refined_layer1_nodes)
                    ),
                    'status': 'refined',
                })
                refined_records.append({
                    **candidate,
                    'refined_layer1_nodes': refined_layer1_nodes,
                    'refined_layer1_snapshot': refined_layer1_snapshot,
                    'refined_layer2_nodes': refined_layer2_nodes,
                    'refined_layer2_stats': copy.deepcopy(
                        self.layer2_estimator.stats
                    ),
                    'region_adjustment': region_adjustment,
                    'shift_coverage': shift_coverage,
                })
            except Exception as exc:
                row['status'] = 'refine_failed'
                row['failure_reason'] = str(exc)
                refinement_failures.append(
                    f"{row['variant_id']}: {row['failure_reason']}"
                )

        if not refined_records:
            raise RuntimeError(
                'layer1_multistart_no_refined_candidate_succeeded: '
                + ' | '.join(refinement_failures[:8])
            )

        refined_records.sort(key=lambda item: (
            -_rank_float(item['diagnostics']['final_layer2_r2']),
            -_rank_float(item['diagnostics']['final_layer1_r2']),
            -int(item['diagnostics']['unique_hop1_motif_count']),
            str(item['diagnostics'].get(
                'final_graph_signature',
                item['diagnostics']['graph_signature'],
            )),
        ))
        winner = refined_records[0]
        winner_row = winner['diagnostics']
        winner_row['selected'] = True
        winner_row['status'] = 'selected'
        winner_layer1_nodes = copy.deepcopy(winner['refined_layer1_nodes'])
        layer1_snapshot = dict(winner['refined_layer1_snapshot'])
        adjust_summary = copy.deepcopy(winner['region_adjustment'])
        layer1_signature_hash = self._layer1_signature_hash(winner_layer1_nodes)
        self.layer1_assigner._last_completion_diagnostics = copy.deepcopy(
            winner['variant'].get('construction', {})
        )
        self.layer1_assigner._last_hop1_adjust_diagnostics = copy.deepcopy(
            adjust_summary
        )
        self.layer1_assigner._last_shift_coverage_audit = copy.deepcopy(
            winner['shift_coverage']
        )

        if eval_output_dir:
            out = Path(eval_output_dir)
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                list(adjust_summary.get('moves', []) or [])
            ).to_csv(out / 'layer1_hop1_adjust_moves.csv', index=False)
        if eval_nmr:
            adjuster.evaluate_with_library(
                nodes=copy.deepcopy(winner_layer1_nodes),
                S_target=S_target,
                output_dir=eval_output_dir,
            )

        if layer2_output_dir:
            nodes = self.layer2_assign(
                nodes=copy.deepcopy(winner_layer1_nodes),
                S_target=S_target,
                E_target=E_target,
                lib_path=eval_lib_path,
                output_dir=layer2_output_dir,
                verbose=bool(verbose),
                emit_summary=False,
            )
            selected_layer2_stats = copy.deepcopy(self.layer2_estimator.stats)
        else:
            nodes = copy.deepcopy(winner['refined_layer2_nodes'])
            selected_layer2_stats = copy.deepcopy(winner['refined_layer2_stats'])
            self.layer2_estimator.stats = copy.deepcopy(selected_layer2_stats)

        if emit_summary:
            print(
                '[Layer1/2] '
                f"candidates={len(variants)} raw_ok={len(candidate_records)} "
                f"refined={len(refined_records)}/{refine_count} "
                f"selected={winner_row['variant_id']} "
                f"layer1_r2={float(layer1_snapshot.get('r2', 0.0)):.4f} "
                f"layer2_r2={float(winner_row['final_layer2_r2']):.4f} "
                f"exact={int(selected_layer2_stats.get('exact_match', 0) or 0)} "
                f"approx={int(selected_layer2_stats.get('approx_match', 0) or 0)} "
                f"missing={int(selected_layer2_stats.get('missing', 0) or 0)} "
                f"elapsed={time.perf_counter() - started_at:.2f}s",
                flush=True,
            )

        csv_output_dir = eval_output_dir or layer2_output_dir
        if csv_output_dir:
            csv_rows: List[Dict[str, Any]] = []
            for row in diagnostics_rows:
                branch = dict(row.get('branch') or {})
                flat_row = {
                    key: value
                    for key, value in row.items()
                    if key != 'branch'
                }
                flat_row.update({
                    'branch_phase': branch.get('phase', ''),
                    'branch_stack_index': branch.get('stack_index', ''),
                    'branch_center_id': branch.get('center_id', ''),
                    'branch_partner_id': branch.get('chosen_partner_id', ''),
                })
                csv_rows.append(flat_row)
            out = Path(csv_output_dir)
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(csv_rows).to_csv(
                out / 'layer1_multistart_candidates.csv',
                index=False,
            )

        build_diagnostics = {
            'build_count': int(len(variants)),
            'construction': copy.deepcopy(
                winner['variant'].get('construction', {})
            ),
            'shift_coverage': copy.deepcopy(winner['shift_coverage']),
            'raw_layer1_r2': float(winner_row['raw_layer1_r2']),
            'final_layer1_r2': float(layer1_snapshot.get('r2', 0.0)),
            'region_adjustment': adjust_summary,
            'final_signature_hash': layer1_signature_hash,
            'multistart': {
                **copy.deepcopy(generation_diagnostics),
                'configured_candidate_count': int(multistart_count),
                'configured_refine_count': int(multistart_refine_count),
                'effective_refine_count': int(refine_count),
                'ranking_policy': (
                    'layer2_r2_desc,layer1_r2_desc,'
                    'unique_hop1_motif_count_desc,graph_signature'
                ),
                'selected_variant_id': str(winner_row['variant_id']),
                'candidates': copy.deepcopy(diagnostics_rows),
            },
            'elapsed_s': float(time.perf_counter() - started_at),
        }

        H_nodes = self._histogram_from_nodes(nodes).detach().clone()
        diff_info = self._compute_difference_spectrum_from_nodes_mu(
            nodes=nodes,
            S_target=S_target,
        )
        diff_info.update({
            'source_stage': 'layer2',
            'spectral_contract': 'template_center_mu',
            'layer1_library_baseline_r2': float(layer1_snapshot.get('r2', 0.0)),
            'layer1_library_baseline_matched_cnt': int(
                layer1_snapshot.get('matched_cnt', 0)
            ),
            'layer1_library_baseline_carbon_cnt': int(
                layer1_snapshot.get('carbon_cnt', 0)
            ),
            'layer1_final_signature_hash': layer1_signature_hash,
            'layer1_build_diagnostics': build_diagnostics,
            'layer2_stats': selected_layer2_stats,
            'layer12_summary': {
                'candidate_count': int(len(variants)),
                'raw_ok_count': int(len(candidate_records)),
                'refined_count': int(len(refined_records)),
                'requested_refine_count': int(refine_count),
                'selected_variant_id': str(winner_row['variant_id']),
                'layer1_r2': float(layer1_snapshot.get('r2', 0.0)),
                'layer2_r2': float(diff_info.get('r2', 0.0)),
                'exact_match': int(selected_layer2_stats.get('exact_match', 0) or 0),
                'approx_match': int(selected_layer2_stats.get('approx_match', 0) or 0),
                'missing': int(selected_layer2_stats.get('missing', 0) or 0),
                'elapsed_s': float(time.perf_counter() - started_at),
            },
        })
        return nodes, H_nodes, diff_info, float(diff_info.get('r2', 0.0))
    def _layer1_signature_hash(self, nodes: Sequence[_NodeV3]) -> str:
        signature = tuple(
            sorted(
                (
                    int(node.su_type),
                    tuple(sorted(int(x) for x in node.hop1_su.elements())),
                )
                for node in nodes
                if int(node.su_type) in self.layer1_assigner._carbon_su_set
            )
        )
        return hashlib.sha1(repr(signature).encode('utf-8')).hexdigest()

    @staticmethod
    def _outer_selected_state_signature(
        nodes: Sequence[_NodeV3],
        H: torch.Tensor,
    ) -> str:
        """Hash the observable state that seeds the next deterministic outer cycle."""
        histogram = tuple(
            int(value)
            for value in torch.clamp(H.detach().cpu(), min=0).long().tolist()
        )
        node_rows = tuple(
            (
                int(getattr(node, 'global_id', -1)),
                int(getattr(node, 'su_type', -1)),
                round(float(getattr(node, 'mu', 0.0) or 0.0), 8),
                round(float(getattr(node, 'pi', 0.0) or 0.0), 8),
                tuple(sorted(
                    int(value)
                    for value in list(getattr(node, 'hop1_ids', []) or [])
                )),
            )
            for node in sorted(
                list(nodes or []),
                key=lambda item: int(getattr(item, 'global_id', -1)),
            )
        )
        return hashlib.sha1(
            repr((histogram, node_rows)).encode('utf-8')
        ).hexdigest()

    @staticmethod
    def _owned_product_node_ids(product: Dict[str, Any]) -> List[int]:
        row = dict(product or {})
        return sorted({
            int(value)
            for value in list(row.get('fixed_source_node_ids', []) or [])
            + [
                int(value)
                for values in dict(row.get('claimed_resource_ids_by_su', {}) or {}).values()
                for value in list(values or [])
            ]
        })

    @staticmethod
    def _stage3_node_state_signature(nodes: Sequence[_NodeV3]) -> List[Dict[str, Any]]:
        return [
            {
                'node_id': int(getattr(node, 'global_id', -1)),
                'su_type': int(getattr(node, 'su_type', -1)),
                'hop1_ids': sorted(int(value) for value in list(getattr(node, 'hop1_ids', []) or [])),
                'fixed_hop1_ids': sorted(
                    int(value) for value in list(getattr(node, 'fixed_hop1_ids', []) or [])
                ),
                'target_hop1_degree': getattr(node, 'target_hop1_degree', None),
                'target_fixed_anchor_count': getattr(node, 'target_fixed_anchor_count', None),
                'special_anchor_mode': str(getattr(node, 'special_anchor_mode', '') or ''),
                'special_anchor_partition': str(
                    getattr(node, 'special_anchor_partition', '') or ''
                ),
            }
            for node in sorted(
                list(nodes or []),
                key=lambda item: int(getattr(item, 'global_id', -1)),
            )
        ]

    @staticmethod
    def _stage3_completion_ownership(ledger: Dict[str, Any]) -> Dict[str, Any]:
        ownership_states = [
            'fixed_topology_core',
            'pending_topology',
            'completed_topology',
            'claimed_closure_resource',
            'remaining_closure_resource',
            'claimed_su24_ring_resource',
            'remaining_su24_ring_resource',
            'remaining_aromatic_donor',
            'support_context',
            'unclassified_or_conflict',
        ]
        states = {state: [] for state in ownership_states}
        for raw in dict(ledger.get('ownership_by_node_id', {}) or {}).values():
            owner = dict(raw or {})
            state = str(owner.get('state', 'unclassified_or_conflict') or 'unclassified_or_conflict')
            states.setdefault(state, []).append(int(owner.get('node_id', -1)))
        states = {key: sorted(set(values)) for key, values in states.items()}
        available = {
            str(key): sorted({int(value) for value in list(values or [])})
            for key, values in dict(ledger.get('available_ids_by_su', {}) or {}).items()
        }
        return {
            'states': states,
            'generic_closure_pool_ids_by_su': {
                str(su_type): list(available.get(str(su_type), []))
                for su_type in (11, 22, 23)
            },
            'su24_ring_resource_ids': list(available.get('24', [])),
            'aromatic_donor_ids_by_su': {
                str(su_type): list(available.get(str(su_type), []))
                for su_type in (10, 12, 13)
            },
            'released_su24_ids': [
                int(value) for value in list(ledger.get('released_su24_ids', []) or [])
            ],
            'released_resource_ids_by_su': {
                str(key): [int(value) for value in list(values or [])]
                for key, values in dict(
                    ledger.get('released_resource_ids_by_su', {}) or {}
                ).items()
            },
            'recycled_su24_to_23_ids': [
                int(value)
                for value in list(ledger.get('recycled_su24_to_23_ids', []) or [])
            ],
            'invariant_errors': [
                dict(item or {}) for item in list(ledger.get('conflicts', []) or [])
            ],
        }

    @staticmethod
    def _stage3_completion_resource_ledger(
        products: Sequence[Dict[str, Any]],
        ownership: Dict[str, Any],
        base_ledger: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the terminal ledger from committed products and final ownership."""
        consumed: Dict[int, set[int]] = {}
        topology_consumed: Dict[int, set[int]] = {}
        residual_consumed: Dict[int, set[int]] = {}
        for raw_product in list(products or []):
            product = dict(raw_product or {})
            phase = str(product.get('phase', '') or '')
            product_type = str(product.get('product_type', '') or '')
            target = (
                residual_consumed
                if phase == 'residual_resource_self_closure'
                or product_type.startswith('residual_')
                else topology_consumed
            )
            for raw_su, raw_ids in dict(
                product.get('claimed_resource_ids_by_su', {}) or {}
            ).items():
                if not str(raw_su).lstrip('-').isdigit():
                    continue
                su_type = int(raw_su)
                node_ids = {int(value) for value in list(raw_ids or [])}
                consumed.setdefault(su_type, set()).update(node_ids)
                target.setdefault(su_type, set()).update(node_ids)

        remaining: Dict[int, set[int]] = {}
        for pool_key in (
            'generic_closure_pool_ids_by_su',
            'aromatic_donor_ids_by_su',
        ):
            for raw_su, raw_ids in dict(ownership.get(pool_key, {}) or {}).items():
                if not str(raw_su).lstrip('-').isdigit():
                    continue
                remaining.setdefault(int(raw_su), set()).update(
                    int(value) for value in list(raw_ids or [])
                )
        remaining.setdefault(24, set()).update(
            int(value)
            for value in list(
                ownership.get('su24_ring_resource_ids', []) or []
            )
        )

        su_types = {
            10, 11, 12, 13, 22, 23, 24,
            *consumed.keys(),
            *remaining.keys(),
        }
        initial = {
            int(su_type): set(consumed.get(int(su_type), set()))
            | set(remaining.get(int(su_type), set()))
            for su_type in su_types
        }
        base = dict(base_ledger or {})
        computed = {
            'initial_counts': {
                str(su_type): len(initial.get(int(su_type), set()))
                for su_type in sorted(su_types)
            },
            'consumed_counts': {
                str(su_type): len(consumed.get(int(su_type), set()))
                for su_type in sorted(su_types)
            },
            'topology_consumed_counts': {
                str(su_type): len(topology_consumed.get(int(su_type), set()))
                for su_type in sorted(su_types)
            },
            'residual_consumed_counts': {
                str(su_type): len(residual_consumed.get(int(su_type), set()))
                for su_type in sorted(su_types)
            },
            'remaining_counts': {
                str(su_type): len(remaining.get(int(su_type), set()))
                for su_type in sorted(su_types)
            },
            'shortage_counts': {
                str(key): int(value)
                for key, value in dict(base.get('shortage_counts', {}) or {}).items()
                if int(value) > 0
            },
            'snapshot_stage': 'stage3_committed',
        }
        # Preserve the authoritative ID-level accounting/provenance captured
        # by Stage3.  Rebuilding the aggregate view from products is useful
        # for reporting, but must not discard recycled 24->23 IDs or the
        # exact remaining pool used by the live allocator.
        for field in (
            'initial_ids_by_su',
            'consumed_ids_by_su',
            'remaining_ids_by_su',
            'released_su24_ids',
            'released_resource_ids_by_su',
            'recycled_su24_to_23_ids',
            'recycled_resource_conversion_ops',
            'invariant_errors',
        ):
            if field in base:
                value = base.get(field)
                if isinstance(value, dict):
                    computed[field] = {
                        str(key): [int(item) for item in list(items or [])]
                        for key, items in value.items()
                    }
                elif isinstance(value, (list, tuple, set)):
                    computed[field] = [int(item) if isinstance(item, (int, np.integer)) or str(item).lstrip('-').isdigit() else item for item in value]
                else:
                    computed[field] = value
        for id_field, count_field in (
            ('initial_ids_by_su', 'initial_counts'),
            ('consumed_ids_by_su', 'consumed_counts'),
            ('remaining_ids_by_su', 'remaining_counts'),
        ):
            ids = computed.get(id_field)
            if isinstance(ids, dict):
                computed[count_field] = {
                    str(key): len(list(values or []))
                    for key, values in ids.items()
                }
        return computed

    def _final_topology_rows(
        self,
        nodes: List[_NodeV3],
        S_target: Optional[torch.Tensor] = None,
        E_target: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        current_node_ids = sorted({
            int(getattr(node, 'global_id', -1)) for node in list(nodes or [])
        })
        snapshot = dict(
            getattr(self, '_final_stage3_completion_diagnostics', {}) or {}
        )
        terminal_snapshot = dict(
            getattr(self, '_final_stage3_terminal_diagnostics', {}) or {}
        )
        snapshot_ids = sorted({
            int(value) for value in list(
                snapshot.get('node_ids', [])
                or getattr(self, '_final_stage3_snapshot_node_ids', [])
                or []
            )
        })
        use_snapshot = bool(
            getattr(self, '_final_stage3_snapshot_authoritative', False)
            and snapshot.get('stage3_complete', False)
            and snapshot_ids
            and snapshot_ids == current_node_ids
        )
        terminal_snapshot_ids = sorted({
            int(value)
            for value in list(terminal_snapshot.get('node_ids', []) or [])
        })
        use_terminal_snapshot = bool(
            not use_snapshot
            and terminal_snapshot
            and terminal_snapshot_ids
            and terminal_snapshot_ids == current_node_ids
        )
        if use_snapshot:
            # This is intentionally a read-only projection of the committed
            # Stage3 ledger.  It prevents post-Layer3 reclassification from
            # reopening already completed phases or losing SU24 ownership.
            fresh_audit = {
                'ok': True,
                'reason': 'stage3_complete',
                'requires_conversion': False,
                'products': [dict(item or {}) for item in list(snapshot.get('products', []) or [])],
                'pending_contracts': [],
                'classification': dict(snapshot.get('classification', {}) or {}),
                'external_assembly': dict(snapshot.get('external_assembly', {}) or {}),
                'ownership_ledger': dict(snapshot.get('ownership_ledger', {}) or {}),
                'resource_ledger': dict(snapshot.get('resource_ledger', {}) or {}),
                'resource_accounting': dict(snapshot.get('resource_accounting', {}) or {'ok': True, 'errors': []}),
                'infeasibility_proof': {},
                'plan': {'phase': 'complete', 'reason': 'stage3_complete'},
                'authoritative_stage3_snapshot': True,
            }
        elif use_terminal_snapshot:
            terminal_plan = dict(terminal_snapshot.get('plan', {}) or {})
            fresh_audit = {
                'ok': False,
                'reason': str(
                    terminal_snapshot.get('stop_reason', '')
                    or terminal_plan.get('reason', '')
                    or 'stage3_incomplete'
                ),
                'requires_conversion': True,
                'products': [
                    dict(item or {})
                    for item in list(terminal_snapshot.get('products', []) or [])
                ],
                'pending_contracts': [
                    dict(item or {})
                    for item in list(
                        terminal_snapshot.get('pending_contracts', []) or []
                    )
                ],
                'classification': dict(
                    terminal_snapshot.get('classification', {}) or {}
                ),
                'external_assembly': dict(
                    terminal_snapshot.get('external_assembly', {}) or {}
                ),
                'ownership_ledger': dict(
                    terminal_snapshot.get('ownership_ledger', {}) or {}
                ),
                'resource_ledger': dict(
                    terminal_snapshot.get('resource_ledger', {}) or {}
                ),
                'resource_accounting': dict(
                    terminal_snapshot.get('resource_accounting', {}) or {}
                ),
                'infeasibility_proof': dict(
                    terminal_snapshot.get('infeasibility_proof', {}) or {}
                ),
                'plan': {
                    **terminal_plan,
                    'phase': str(
                        terminal_snapshot.get('phase', '')
                        or terminal_plan.get('phase', '')
                    ),
                },
                'generation_id': int(
                    terminal_snapshot.get('generation_id', 0) or 0
                ),
                'authoritative_stage3_terminal_snapshot': True,
            }
        else:
            fresh_audit = audit_stage3_topology(
                H=self._histogram_from_nodes(list(nodes or [])).detach().cpu().long(),
                nodes=list(nodes or []),
                fixed_partition_meta=getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {},
                S_target=S_target,
                E_target=E_target,
                allow_conversions=False,
                enable_spectrum_rebalance=False,
            )
        classification_dict = dict(fresh_audit.get('classification', {}) or {})
        assembly_dict = dict(fresh_audit.get('external_assembly', {}) or {})
        public_rows = [
            dict(item or {}) for item in list(fresh_audit.get('products', []) or [])
        ]
        public_pending = [
            dict(item or {})
            for item in list(fresh_audit.get('pending_contracts', []) or [])
        ]
        public_summary = {
            'ok': bool(fresh_audit.get('ok', False)),
            'reason': str(fresh_audit.get('reason', 'fresh_stage3_audit_failed') or ''),
            'topology_plan_feasible': bool(fresh_audit.get('ok', False)),
            'closed_topology_count': int(len(public_rows)),
            'final_topology_count': int(len(public_rows)),
            'pending_closure_total_count': int(len(public_pending)),
            'unclosed_candidates': list(public_pending),
            'resource_ledger': dict(fresh_audit.get('resource_ledger', {}) or {}),
        }
        completion_diagnostics = {
            'stage3_complete': bool(fresh_audit.get('ok', False)),
            'generation_id': 0,
            'products': list(public_rows),
            'ownership_ledger': dict(fresh_audit.get('ownership_ledger', {}) or {}),
            'residual_infeasibility_proof': dict(
                fresh_audit.get('infeasibility_proof', {}) or {}
            ),
        }
        terminal_diagnostics = (
            {}
            if bool(fresh_audit.get('ok', False))
            else {
                'stop_reason': str(fresh_audit.get('reason', '') or ''),
                'phase': str(
                    dict(fresh_audit.get('plan', {}) or {}).get('phase', '') or ''
                ),
                'generation_id': 0,
                'infeasibility_proof': dict(
                    fresh_audit.get('infeasibility_proof', {}) or {}
                ),
            }
        )
        residual_infeasibility_proof = dict(
            completion_diagnostics.get('residual_infeasibility_proof', {}) or {}
        )
        completion_matches = bool(completion_diagnostics.get('stage3_complete', False))
        terminal_matches = bool(terminal_diagnostics)
        stage3_snapshot_matches = bool(completion_matches or terminal_matches)
        infeasibility_proof = (
            dict(terminal_diagnostics.get('infeasibility_proof', {}) or {})
            if terminal_matches else {}
        )
        completion_ownership: Dict[str, Any] = dict(
            fresh_audit.get('ownership_ledger', {}) or {}
        )
        completion_coverage: Dict[str, Any] = {}
        if completion_matches:
            public_rows = [
                dict(item or {})
                for item in list(completion_diagnostics.get('products', []) or [])
            ]
            public_pending = []
            completion_ownership = dict(
                completion_diagnostics.get('ownership_ledger', {}) or {}
            )
            completion_resource_ledger = dict(
                fresh_audit.get('resource_ledger', {}) or {}
            )
            public_summary.update({
                'ok': True,
                'reason': 'stage3_complete',
                'topology_plan_feasible': True,
                'closed_topology_count': int(len(public_rows)),
                'final_topology_count': int(len(public_rows)),
                'pending_closure_total_count': 0,
                'unclosed_candidates': [],
                'resource_ledger': completion_resource_ledger,
            })
            states = dict(completion_ownership.get('states', {}) or {})
            all_ids = {int(getattr(node, 'global_id', -1)) for node in list(nodes or [])}
            completed_ids_for_coverage = {
                int(value)
                for state in (
                    'completed_topology',
                    'claimed_closure_resource',
                    'claimed_su24_ring_resource',
                )
                for value in list(states.get(state, []) or [])
            }
            pending_ids_for_coverage = {
                int(value) for value in list(states.get('pending_topology', []) or [])
            }
            resource_ids_for_coverage = {
                int(value)
                for state in (
                    'remaining_closure_resource',
                    'remaining_su24_ring_resource',
                    'remaining_aromatic_donor',
                )
                for value in list(states.get(state, []) or [])
            }
            fixed_ids_for_coverage = {
                int(value)
                for value in list(states.get('fixed_topology_core', []) or [])
            }
            support_ids_for_coverage = {
                int(value)
                for value in list(states.get('support_context', []) or [])
            }
            conflict_ids_for_coverage = {
                int(value)
                for value in list(states.get('unclassified_or_conflict', []) or [])
            }
            completion_coverage = {
                'total_node_count': int(len(all_ids)),
                'completed_node_count': int(len(completed_ids_for_coverage)),
                'pending_node_count': int(len(pending_ids_for_coverage)),
                'remaining_pool_node_count': int(len(resource_ids_for_coverage)),
                'conflict_node_count': int(len(conflict_ids_for_coverage)),
                'conflict_node_ids': sorted(conflict_ids_for_coverage),
                'covered_node_count': int(sum(len(list(values or [])) for values in states.values())),
                'uncovered_node_ids': sorted(
                    all_ids
                    - completed_ids_for_coverage
                    - pending_ids_for_coverage
                    - resource_ids_for_coverage
                    - fixed_ids_for_coverage
                    - support_ids_for_coverage
                    - conflict_ids_for_coverage
                ),
            }
        terminal_residual_complete = bool(completion_matches)
        final_reason = str(
            'stage3_complete'
            if terminal_residual_complete
            else fresh_audit.get('reason', 'fresh_stage3_audit_failed')
        )
        cls_summary = dict(classification_dict.get('summary', {}) or {})
        covered_product_node_ids = {
            int(node_id)
            for product in list(public_rows or [])
            for node_id in InversePipelineV3._owned_product_node_ids(dict(product or {}))
        }
        rejects = [
            dict(record or {})
            for record in list(classification_dict.get('classification_rejects', []) or [])
            if int(dict(record or {}).get('node_id', dict(record or {}).get('global_id', -1)))
            not in covered_product_node_ids
        ]
        unclassified_non_resource = [
            dict(rec or {})
            for rec in list(classification_dict.get('unclassified_nodes', []) or [])
            if not bool(dict(rec or {}).get('resource_pool_member', False))
            and int(dict(rec or {}).get('node_id', dict(rec or {}).get('global_id', -1)))
            not in covered_product_node_ids
        ]
        for rec in unclassified_non_resource:
            rec.setdefault('reason', 'unclassified_non_resource')
        summary = {
            'ok': bool(terminal_residual_complete),
            'topology_plan_feasible': bool(terminal_residual_complete),
            'closed_topology_count': int(public_summary.get('closed_topology_count', 0)),
            'final_topology_count': int(public_summary.get('final_topology_count', 0)),
            'pending_closure_total_count': int(public_summary.get('pending_closure_total_count', 0)),
            'unclosed_candidates': list(public_pending),
            'classification_rejects': rejects + unclassified_non_resource,
            'diagnostics': {
                'reason': str(final_reason),
                'solver_reason': str(final_reason),
                'infeasibility_proof': dict(infeasibility_proof),
                'residual_infeasibility_proof': dict(residual_infeasibility_proof),
                'class_counts': dict(cls_summary.get('class_counts', {}) or {}),
                'family_counts': dict(cls_summary.get('family_counts', {}) or {}),
                'resource_pool': dict(classification_dict.get('resource_pool', {}) or {}),
                'flex_bounds': dict(assembly_dict.get('flex_bounds', {}) or {}),
                'flex_counts': dict(assembly_dict.get('flex_counts', {}) or {}),
                'resource_ledger': dict(public_summary.get('resource_ledger', {}) or {}),
                'assembly': dict(assembly_dict.get('diagnostics', {}) or {}),
                'flex_target_plan': dict(assembly_dict.get('flex_target_plan', {}) or {}),
                'resource_demand': dict(assembly_dict.get('resource_demand', {}) or {}),
                'protected_resource_counts': dict(assembly_dict.get('protected_resource_counts', {}) or {}),
                'referenced_resource_counts': dict(assembly_dict.get('referenced_resource_counts', {}) or {}),
                'node_coverage': (
                    dict(completion_coverage)
                    if completion_matches
                    else dict(dict(assembly_dict.get('diagnostics', {}) or {}).get('node_coverage', {}) or {})
                ),
                'ownership_ledger': (
                    dict(completion_ownership)
                    if completion_ownership
                    else dict(dict(assembly_dict.get('diagnostics', {}) or {}).get('ownership_ledger', {}) or {})
                ),
                'stage3_generation_id': int(
                    completion_diagnostics.get('generation_id', 0) or 0
                ) if completion_matches else int(
                    terminal_diagnostics.get('generation_id', 0) or 0
                ) if terminal_matches else 0,
                'stage3_terminal_stop_reason': str(
                    terminal_diagnostics.get('stop_reason', '') or ''
                ),
                'stage3_terminal_phase': str(
                    terminal_diagnostics.get('phase', '') or ''
                ),
                'stage3_terminal_snapshot_match': bool(stage3_snapshot_matches),
                'fresh_stage3_audit': {
                    'ok': bool(fresh_audit.get('ok', False)),
                    'reason': str(fresh_audit.get('reason', '') or ''),
                    'authoritative_stage3_snapshot': bool(
                        fresh_audit.get('authoritative_stage3_snapshot', False)
                    ),
                    'requires_conversion': bool(
                        fresh_audit.get('requires_conversion', False)
                    ),
                    'resource_shortage': dict(
                        fresh_audit.get('resource_shortage', {}) or {}
                    ),
                    'resource_accounting': dict(
                        fresh_audit.get('resource_accounting', {}) or {}
                    ),
                },
            },
        }
        coverage = dict(summary['diagnostics'].get('node_coverage', {}) or {})
        if not coverage:
            completed_ids = {
                int(node_id)
                for row in public_rows
                for node_id in InversePipelineV3._owned_product_node_ids(dict(row or {}))
            }
            pending_ids = {
                int(node_id)
                for row in public_pending
                for node_id in list(
                    dict(row or {}).get('claimed_source_node_ids', []) or []
                )
            }
            resource_ledger = dict(summary['diagnostics'].get('resource_ledger', {}) or {})
            ownership_states = dict(
                dict(summary['diagnostics'].get('ownership_ledger', {}) or {}).get('states', {}) or {}
            )
            resource_ids = {
                int(node_id)
                for state in (
                    'remaining_closure_resource',
                    'remaining_su24_ring_resource',
                    'remaining_aromatic_donor',
                )
                for node_id in list(ownership_states.get(state, []) or [])
            }
            all_ids = {int(getattr(node, 'global_id', -1)) for node in list(nodes or [])}
            coverage = {
                'total_node_count': int(len(all_ids)),
                'completed_node_count': int(len(completed_ids)),
                'pending_node_count': int(len(pending_ids)),
                'remaining_pool_node_count': int(len(resource_ids)),
                'uncovered_node_ids': sorted(all_ids - completed_ids - pending_ids - resource_ids),
            }
            summary['diagnostics']['node_coverage'] = coverage
        summary['coverage_valid'] = not bool(list(coverage.get('uncovered_node_ids', []) or []))
        summary['classification_coverage_valid'] = not bool(
            list(coverage.get('conflict_node_ids', []) or [])
        )
        completed_ids = {
            int(node_id)
            for row in public_rows
            for node_id in InversePipelineV3._owned_product_node_ids(dict(row or {}))
        }
        pending_ids = {
            int(node_id)
            for row in public_pending
            for node_id in list(
                dict(row or {}).get('claimed_source_node_ids', []) or []
            )
        }
        resource_ledger = dict(summary['diagnostics'].get('resource_ledger', {}) or {})
        ownership_states = dict(
            dict(summary['diagnostics'].get('ownership_ledger', {}) or {}).get('states', {}) or {}
        )
        resource_ids = {
            int(node_id)
            for state in (
                'remaining_closure_resource',
                'remaining_su24_ring_resource',
                'remaining_aromatic_donor',
            )
            for node_id in list(ownership_states.get(state, []) or [])
        }
        fixed_topology_ids = {
            int(node_id)
            for node_id in list(ownership_states.get('fixed_topology_core', []) or [])
        }
        support_context_ids = {
            int(node_id)
            for node_id in list(ownership_states.get('support_context', []) or [])
        }
        ownership_conflict_ids = {
            int(node_id)
            for node_id in list(
                ownership_states.get('unclassified_or_conflict', []) or []
            )
        }
        invariant_conflicts: List[Dict[str, Any]] = []
        for left_name, left_ids, right_name, right_ids in (
            ('completed_topology', completed_ids, 'pending_topology', pending_ids),
            ('completed_topology', completed_ids, 'available_resource', resource_ids),
            ('pending_topology', pending_ids, 'available_resource', resource_ids),
        ):
            overlap = sorted(set(left_ids) & set(right_ids))
            if overlap:
                invariant_conflicts.append({
                    'reason': 'identity_overlap',
                    'states': [left_name, right_name],
                    'node_ids': overlap,
                })
        for row in public_rows:
            source_ids = InversePipelineV3._owned_product_node_ids(dict(row or {}))
            gid_ids = sorted({
                int(value)
                for value in re.findall(r'\d+', str(dict(row or {}).get('node_id_notation', '') or ''))
            })
            if source_ids != gid_ids:
                invariant_conflicts.append({
                    'reason': 'product_notation_source_mismatch',
                    'node_id_notation': str(dict(row or {}).get('node_id_notation', '')),
                    'owned_node_ids': source_ids,
                    'gid_node_ids': gid_ids,
                })
            product_claimed_ids = {
                int(value)
                for values in dict(
                    dict(row or {}).get('claimed_resource_ids_by_su', {}) or {}
                ).values()
                for value in list(values or [])
            }
            for slot in (
                list(dict(row or {}).get('mainline_slots', []) or [])
                + list(dict(row or {}).get('branch_slots', []) or [])
            ):
                assigned_ids = [
                    int(value)
                    for value in list(dict(slot or {}).get('assigned_resource_ids', []) or [])
                ]
                if len(assigned_ids) != len(set(assigned_ids)):
                    invariant_conflicts.append({
                        'reason': 'duplicate_resource_within_slot',
                        'node_id_notation': str(dict(row or {}).get('node_id_notation', '')),
                        'slot_id': str(dict(slot or {}).get('slot_id', '')),
                        'node_ids': assigned_ids,
                    })
                unexpected_ids = sorted(set(assigned_ids) - product_claimed_ids)
                if unexpected_ids:
                    invariant_conflicts.append({
                        'reason': 'slot_resource_outside_product_ownership',
                        'node_id_notation': str(dict(row or {}).get('node_id_notation', '')),
                        'slot_id': str(dict(slot or {}).get('slot_id', '')),
                        'node_ids': unexpected_ids,
                    })
        for row in public_pending:
            if str(dict(row or {}).get('pending_kind', '') or '') in {
                'internal_su24_resource_conversion',
                'internal_resource_conversion',
            }:
                # This is an internal resource action, not a topology source
                # contract.  Its display gid identifies the SU24 donor, but
                # it intentionally has no claimed topology source IDs.
                continue
            claimed_source_ids = sorted({
                int(value)
                for value in list(
                    dict(row or {}).get('claimed_source_node_ids', [])
                    or []
                )
            })
            gid_ids = sorted({int(value) for value in re.findall(r'\d+', str(dict(row or {}).get('real_gid', '') or ''))})
            infeasibility_proof = dict(dict(row or {}).get('infeasibility_proof', {}) or {})
            suppressed_support_ids = sorted({
                int(value)
                for value in list(infeasibility_proof.get('suppressed_support_node_ids', []) or [])
                if int(value) in set(claimed_source_ids)
            })
            expected_gid_ids = sorted(
                set(claimed_source_ids) - set(suppressed_support_ids)
            )
            missing_gid_ids = sorted(set(expected_gid_ids) - set(gid_ids))
            extra_gid_ids = sorted(set(gid_ids) - set(claimed_source_ids))
            if missing_gid_ids or extra_gid_ids:
                invariant_conflicts.append({
                    'reason': 'pending_gid_source_mismatch',
                    'contract_id': int(dict(row or {}).get('contract_id', dict(row or {}).get('unit_id', -1)) or -1),
                    'claimed_source_node_ids': claimed_source_ids,
                    'gid_node_ids': gid_ids,
                    'suppressed_support_node_ids': suppressed_support_ids,
                    'missing_gid_node_ids': missing_gid_ids,
                    'extra_gid_node_ids': extra_gid_ids,
                })
        identity_by_node = {
            **{int(value): 'completed_topology' for value in completed_ids},
            **{int(value): 'pending_topology' for value in pending_ids},
            **{int(value): 'available_resource' for value in resource_ids},
            **{int(value): 'fixed_topology_core' for value in fixed_topology_ids},
            **{int(value): 'support_context' for value in support_context_ids},
            **{
                int(value): 'unclassified_or_conflict'
                for value in ownership_conflict_ids
            },
        }
        for group in list(classification_dict.get('atomic_groups', []) or []):
            group_ids = [int(value) for value in list(dict(group or {}).get('node_ids', []) or [])]
            states = {identity_by_node.get(int(value), 'other') for value in group_ids}
            if 'other' in states or 'unclassified_or_conflict' in states:
                invariant_conflicts.append({
                    'reason': 'atomic_pair_unowned',
                    'group_id': int(dict(group or {}).get('group_id', -1) or -1),
                    'node_ids': group_ids,
                    'states': sorted(states),
                })
            elif len(states) > 1:
                invariant_conflicts.append({
                    'reason': 'atomic_pair_split',
                    'group_id': int(dict(group or {}).get('group_id', -1) or -1),
                    'node_ids': group_ids,
                    'states': sorted(states),
                })
        summary['diagnostics']['topology_invariant_conflicts'] = invariant_conflicts
        if invariant_conflicts:
            summary['ok'] = False
            summary['coverage_valid'] = False
            if not dict(summary['diagnostics'].get('infeasibility_proof', {}) or {}):
                summary['diagnostics']['solver_reason'] = 'topology_invariant_violation'
        su_identity_audit = self._audit_final_su_identity(nodes)
        summary['diagnostics']['su_identity_audit'] = su_identity_audit
        if not bool(su_identity_audit.get('ok', False)):
            summary['ok'] = False
            summary['coverage_valid'] = False
            summary['diagnostics']['solver_reason'] = 'final_su_identity_mismatch'
        return public_rows, summary

    @staticmethod
    def _compact_topology_export_rows(
        topology_rows: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return [
            {
                'actual_su': str(row.get('topology_notation', '')),
                'gid': str(row.get('node_id_notation', '')),
                'product_type': str(row.get('product_type', '')),
                'required_slots_satisfied': bool(row.get('closed', False)),
                'source_fragment_ids': json.dumps(
                    row.get('source_fragment_ids', []), ensure_ascii=False
                ),
                'fixed_source_node_ids': json.dumps(
                    row.get('fixed_source_node_ids', []), ensure_ascii=False
                ),
                'claimed_resource_ids': json.dumps(
                    row.get('claimed_resource_ids_by_su', {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            for row in (dict(item or {}) for item in list(topology_rows or []))
        ]

    @staticmethod
    def _compact_pending_export_rows(
        pending_rows: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        compact_rows: List[Dict[str, Any]] = []
        for row in (dict(item or {}) for item in list(pending_rows or [])):
            claimed_source_ids = row.get('claimed_source_node_ids', []) or []
            real = str(
                row.get('real', '')
                or row.get('real_topology_partial', '')
                or row.get('expected_topology', '')
                or ''
            )
            real_gid = str(
                row.get('real_gid', '')
                or row.get('real_node_partial', '')
                or row.get('node_id', '')
                or ''
            )
            compact_rows.append({
                'actual_su': real,
                'gid': real_gid,
                'pending_kind': str(row.get('pending_kind', '')),
                'phase': str(row.get('phase', '')),
                'contract_id': int(row.get('contract_id', -1) or -1),
                'source_fragment_ids': json.dumps(
                    row.get('source_fragment_ids', []), ensure_ascii=False
                ),
                'claimed_source_node_ids': json.dumps(
                    claimed_source_ids,
                    ensure_ascii=False,
                ),
                'required_counts': json.dumps(
                    row.get('required_counts', {}), ensure_ascii=False, sort_keys=True
                ),
                'available_counts': json.dumps(
                    row.get('available_counts', {}), ensure_ascii=False, sort_keys=True
                ),
                'reason': str(row.get('reason', '')),
            })
        return compact_rows

    @staticmethod
    def _topology_ownership_rows(
        nodes: List[_NodeV3],
        topology_rows: List[Dict[str, Any]],
        pending_rows: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        completed_by_node: Dict[int, Dict[str, Any]] = {}
        pending_by_node: Dict[int, Dict[str, Any]] = {}
        for row in list(topology_rows or []):
            product = dict(row or {})
            for node_id in InversePipelineV3._owned_product_node_ids(product):
                completed_by_node[int(node_id)] = {
                    'topology_id': str(product.get('node_id_notation', '')),
                    'contract_id': '',
                    'fragment_id': str(product.get('source_fragment_ids', '')),
                    'fragment_class': str(product.get('product_type', '')),
                    'role': 'completed_topology',
                }
        for row in list(pending_rows or []):
            for node_id in list(
                dict(row or {}).get('claimed_source_node_ids', []) or []
            ):
                pending_by_node[int(node_id)] = {
                    'topology_id': str(dict(row or {}).get('real_gid', '')),
                    'contract_id': str(dict(row or {}).get('contract_id', '')),
                    'fragment_id': str(dict(row or {}).get('source_fragment_ids', '')),
                    'fragment_class': str(
                        dict(row or {}).get('source_fragment_classes', [])
                        or dict(row or {}).get('pending_kind', '')
                    ),
                    'role': 'pending_topology',
                }
        diagnostics = dict(summary.get('diagnostics', {}) or {})
        ownership = dict(diagnostics.get('ownership_ledger', {}) or {})
        states = dict(ownership.get('states', {}) or {})
        state_by_node = {
            int(node_id): str(state)
            for state, node_ids in states.items()
            for node_id in list(node_ids or [])
        }
        rows: List[Dict[str, Any]] = []
        conflicts: List[Dict[str, Any]] = [
            dict(item or {}) for item in list(ownership.get('invariant_errors', []) or [])
        ]
        for node in sorted(list(nodes or []), key=lambda item: int(getattr(item, 'global_id', -1))):
            node_id = int(getattr(node, 'global_id', -1))
            state = str(state_by_node.get(node_id, 'unclassified_or_conflict'))
            if state in {
                'completed_topology',
                'claimed_closure_resource',
                'claimed_su24_ring_resource',
            }:
                meta = completed_by_node.get(node_id, {'role': state})
            elif state == 'pending_topology':
                meta = pending_by_node.get(node_id, {'role': state})
            else:
                meta = {'role': state}
            meta = dict(meta or {})
            rows.append({
                'node_id': node_id,
                'su_type': int(getattr(node, 'su_type', -1)),
                'state': str(state),
                'topology_id': str(meta.get('topology_id', '')),
                'contract_id': str(meta.get('contract_id', '')),
                'fragment_id': str(meta.get('fragment_id', '')),
                'fragment_class': str(meta.get('fragment_class', '')),
                'role': str(meta.get('role', state)),
                'slot_id': str(meta.get('slot_id', '')),
                'resource_pool': bool(state in {
                    'remaining_closure_resource',
                    'remaining_su24_ring_resource',
                    'remaining_aromatic_donor',
                }),
            })
        return rows, conflicts

    def _print_final_topology_specs(
        self,
        nodes: List[_NodeV3],
        limit: int = 20,
        S_target: Optional[torch.Tensor] = None,
        E_target: Optional[torch.Tensor] = None,
    ) -> None:
        try:
            rows, summary = self._final_topology_rows(
                nodes,
                S_target=S_target,
                E_target=E_target,
            )
            solver_reason = str(
                dict(summary.get('diagnostics', {}) or {}).get('solver_reason', '')
            )
            print(
                "[Final Allocation] "
                f"ok={bool(summary.get('ok', False))} "
                f"reason={solver_reason}"
            )
            print(
                "[Final Topology] "
                f"ok={bool(summary.get('ok', False))} "
                f"status={solver_reason} "
                f"closed={int(summary.get('closed_topology_count', 0))} "
                f"final_specs={int(summary.get('final_topology_count', 0))} "
                f"pending={int(summary.get('pending_closure_total_count', 0))}"
            )
            ledger = dict(dict(summary.get('diagnostics', {}) or {}).get('resource_ledger', {}) or {})
            consumed_counts = {
                str(k): int(v)
                for k, v in dict(ledger.get('consumed_counts', {}) or {}).items()
            }
            topology_consumed_counts = {
                str(k): int(v)
                for k, v in dict(ledger.get('topology_consumed_counts', {}) or {}).items()
            }
            residual_consumed_counts = {
                str(k): int(v)
                for k, v in dict(ledger.get('residual_consumed_counts', {}) or {}).items()
            }
            remaining_counts = {
                str(k): int(v)
                for k, v in dict(ledger.get('remaining_counts', {}) or {}).items()
            }
            shortage_counts = {
                str(k): int(v)
                for k, v in dict(ledger.get('shortage_counts', {}) or {}).items()
                if int(v) > 0
            }
            initial_counts = {
                str(k): int(v)
                for k, v in dict(ledger.get('initial_counts', {}) or {}).items()
            }
            resource_keys = set(initial_counts.keys()) | set(consumed_counts.keys()) | set(remaining_counts.keys())

            def _resource_sort_key(value: str) -> Tuple[int, Any]:
                try:
                    return (0, int(value))
                except Exception:
                    return (1, str(value))

            ordered_resource_keys = sorted(resource_keys, key=_resource_sort_key)

            def _format_resource_counts(counts: Dict[str, int], keys: List[str]) -> str:
                if not keys:
                    return "{}"
                return "{" + ", ".join(
                    f"{key}: {int(counts.get(str(key), 0) or 0)}"
                    for key in list(keys or [])
                ) + "}"

            print(
                "  [Resource Ledger] "
                f"initial_total={int(sum(initial_counts.values()))} "
                f"consumed_total={int(sum(consumed_counts.values()))} "
                f"remaining_total={int(sum(remaining_counts.values()))} "
                f"shortage_total={int(sum(shortage_counts.values()))}"
            )
            print(f"    initial_counts={_format_resource_counts(initial_counts, ordered_resource_keys)}")
            print(f"    consumed_counts={_format_resource_counts(consumed_counts, ordered_resource_keys)}")
            print(f"    topology_consumed_counts={_format_resource_counts(topology_consumed_counts, ordered_resource_keys)}")
            print(f"    residual_consumed_counts={_format_resource_counts(residual_consumed_counts, ordered_resource_keys)}")
            print(f"    remaining_counts={_format_resource_counts(remaining_counts, ordered_resource_keys)}")
            print(f"    shortage_counts={_format_resource_counts(shortage_counts, sorted(shortage_counts.keys(), key=_resource_sort_key))}")
            print(f"    snapshot_stage={str(ledger.get('snapshot_stage', ''))}")
            shown = rows[:max(0, int(limit))]
            for idx, row in enumerate(shown, start=1):
                print(
                    "  [TopologySpec] "
                    f"#{int(idx)} "
                    f"topology={row.get('topology_notation', '')} "
                    f"node_ids={row.get('node_id_notation', '')} "
                    f"source_fragment_ids={row.get('source_fragment_ids', [])} "
                    f"fixed_source_node_ids={row.get('fixed_source_node_ids', [])} "
                    f"claimed_resource_ids_by_su={row.get('claimed_resource_ids_by_su', {})}"
                )
            if len(rows) > len(shown):
                print(f"  [TopologySpec] ... {len(rows) - len(shown)} more")
            unclosed = list(summary.get('unclosed_candidates', []) or [])
            if unclosed:
                print(f"  [Final Topology Pending] showing={min(8, len(unclosed))}/{len(unclosed)}")
                for idx, row in enumerate(unclosed[:8]):
                    rec = dict(row or {})
                    print(
                        f"    [{idx}] real={rec.get('real', '')} "
                        f"real_gid={rec.get('real_gid', '')}"
                    )
            rejects = list(summary.get('classification_rejects', []) or [])
            if rejects:
                print(f"  [Final Topology Unrecognized] showing={min(8, len(rejects))}/{len(rejects)}")
                for idx, row in enumerate(rejects[:8]):
                    rec = dict(row or {})
                    print(
                        f"    [{idx}] gid={rec.get('global_id', '')} "
                        f"su={rec.get('su_type', '')} "
                        f"reason={rec.get('reason', '')}"
                    )
        except Exception as e:
            print(f"[Final Topology] Failed to summarize topology specs: {e}")

    def _stage3_topology_signature(self,
                                   H: torch.Tensor,
                                   assembly: Dict[str, Any]) -> Tuple[Any, ...]:
        H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
        h_part = tuple(
            int(H_cpu[int(su)].item()) if int(su) < int(H_cpu.numel()) else 0
            for su in (10, 11, 12, 13, 22, 23, 24)
        )
        flex_counts = dict((assembly or {}).get('flex_counts', {}) or {})
        pending_count = len(list((assembly or {}).get('pending_units', []) or []))
        diag = dict((assembly or {}).get('diagnostics', {}) or {})
        total_shortage = dict(diag.get('total_shortage', {}) or {})
        residual = dict(diag.get('residual_resource_counts', {}) or {})
        shortage_part = tuple(
            int(total_shortage.get(str(su), total_shortage.get(int(su), 0)) or 0)
            for su in (11, 22, 23, 24)
        )
        residual_part = tuple(
            int(residual.get(str(su), residual.get(int(su), 0)) or 0)
            for su in (11, 22, 23, 24)
        )
        return (
            *h_part,
            int(flex_counts.get('final', 0) or 0),
            int(diag.get('explicit_flex_count', flex_counts.get('explicit_total', flex_counts.get('explicit', 0))) or 0),
            int(pending_count),
            shortage_part,
            residual_part,
            str((assembly or {}).get('reason', '')),
        )

    @staticmethod
    def _stage3_metrics_scalar(metrics: Dict[str, Any]) -> int:
        rec = dict(metrics or {})
        return (
            int(rec.get('pending_count', 0) or 0) +
            int(rec.get('flex_deficit', 0) or 0) +
            int(rec.get('resource_deficit_total', 0) or 0)
        )

    def _validate_stage3_candidate(self,
                                   nodes: List[_NodeV3],
                                   H: torch.Tensor,
                                   S_target: torch.Tensor,
                                   E_target: torch.Tensor,
                                   tag: str) -> Tuple[bool, str]:
        meta_before_sync = copy.deepcopy(getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {})
        try:
            graph_ok, graph_errors = self.layer1_assigner.validate_graph_consistency(
                nodes=nodes,
                H=H,
                E_target=E_target,
                verbose=False,
            )
        except Exception as e:
            print(f"[{tag}] 图校验失败: {e}")
            return False, 'graph_check_exception'
        if not bool(graph_ok) or graph_errors:
            print(f"[{tag}] 回退: 图结构校验未通过，errors={len(graph_errors)}")
            try:
                self.layer1_assigner._print_consistency_errors(graph_errors, prefix=tag, limit=6)
            except Exception:
                pass
            return False, 'graph_invalid'
        try:
            self._sync_fixed_partition_meta_from_nodes(nodes)
        except Exception as e:
            self._set_fixed_partition_meta(meta_before_sync)
            print(f"[{tag}] 固定锚点元数据同步失败: {e}")
            return False, 'meta_sync_exception'
        try:
            hist_diag = self.layer4_adjuster._evaluate_required_hist_constraints(
                H,
                E_target,
                S_target=S_target,
            )
        except Exception as e:
            self._set_fixed_partition_meta(meta_before_sync)
            print(f"[{tag}] 直方图校验失败: {e}")
            return False, 'hist_check_exception'
        if not bool(hist_diag.get('ok', False)):
            self._set_fixed_partition_meta(meta_before_sync)
            reason = str(hist_diag.get('reason', 'hist_invalid'))
            print(f"[{tag}] 回退: 直方图/固定连接校验未通过，reason={reason}")
            return False, reason
        try:
            classification = classify_topology_units(nodes)
            projection_preflight = preflight_fragment_view_projection(
                nodes,
                classification,
            )
            if not bool(projection_preflight.get('ok', False)):
                self._set_fixed_partition_meta(meta_before_sync)
                print(
                    f"[{tag}] 回退: Stage3 fragment 投影不可物化，"
                    f"reason={projection_preflight.get('reason', '')}"
                )
                return False, str(
                    projection_preflight.get(
                        'reason',
                        'stage3_fragment_projection_invalid',
                    )
                )
            flex_audit = _compute_flex_audit(
                nodes,
                classification,
                enforce=True,
            )
            route_preflight = stage3_fixed_route_preflight(
                nodes,
                classification,
                flex_audit,
            )
        except Exception as e:
            self._set_fixed_partition_meta(meta_before_sync)
            print(f"[{tag}] Stage3 固定路径预检失败: {e}")
            return False, 'stage3_structural_preflight_exception'
        if not bool(route_preflight.get('ok', False)):
            self._set_fixed_partition_meta(meta_before_sync)
            failed_ids = sorted(
                int(value)
                for value in dict(
                    route_preflight.get('failures_by_fragment_id', {}) or {}
                )
            )
            print(
                f"[{tag}] 回退: Stage3 固定路径不可物化，fragments={failed_ids}"
            )
            return False, str(
                route_preflight.get(
                    'reason',
                    'stage3_fixed_route_materialization_invalid',
                )
            )
        return True, 'ok'

    def _run_topology_stage3_resource_cycles(self,
                                             nodes: List[_NodeV3],
                                             H: torch.Tensor,
                                             current_diff: Dict[str, object],
                                             S_target: torch.Tensor,
                                             E_target: torch.Tensor,
                                             eval_lib_path: Optional[str],
                                             layer1_runtime_cfg: Dict[str, Any],
                                             outer_idx: int,
                                             flex_max_primary_adjustments: int = 10,
                                             external_max_primary_adjustments: int = 10,
                                             internal_max_primary_adjustments: int = 30,
                                             max_h_compensation_ops_per_adjustment: int = 8,
                                             residual_flex_preferred_23: int = 3,
                                             residual_flex_min_23: int = 2,
                                             residual_side_preferred_23: int = 2,
                                             residual_side_min_23: int = 1,
                                             h_tolerance: float = 0.08,
                                             enable_post_external_spectrum: bool = True,
                                             spectrum_active_through_cycle: int = 20,
                                             spectrum_candidate_limit: int = 8,
                                             spectrum_max_candidate_trials: int = 4,
                                             spectrum_max_primary_adjustments: int = 10,
                                             spectrum_min_r2_gain: float = 0.0005,
                                             verbose: bool = False,
                                             iteration_offset: int = 0,
                                             output_dir: Optional[str] = None) -> Tuple[List[_NodeV3], torch.Tensor, Dict[str, object], Dict[str, Any]]:
        baseline_diff = dict(current_diff or {})
        if (
            str(baseline_diff.get('source_stage', '') or '') != 'layer2'
            or str(baseline_diff.get('spectral_contract', '') or '') != 'template_center_mu'
        ):
            raise ValueError('stage3_requires_saved_layer2_baseline')

        def rebuild_fn(H_seed: torch.Tensor,
                       phase: str,
                       cycle_idx: int,
                       nodes_seed: Optional[List[_NodeV3]] = None,
                       ) -> Tuple[List[_NodeV3], torch.Tensor, Dict[str, Any], float]:
            layer1_stage3_cfg = dict(layer1_runtime_cfg)

            cand_nodes, cand_H, cand_diff, cand_r2 = self._run_layer12_cycle(
                H_seed=H_seed.detach().clone(),
                S_target=S_target,
                E_target=E_target,
                eval_lib_path=eval_lib_path,
                layer1_runtime_cfg=layer1_stage3_cfg,
                eval_nmr=False,
                identity_seed_nodes=nodes_seed,
                emit_summary=False,
            )
            cand_H = self._histogram_from_nodes(cand_nodes).detach().clone()
            cand_diff = dict(cand_diff or {})
            cand_diff['source_stage'] = 'layer2'
            cand_diff['spectral_contract'] = 'template_center_mu'
            cand_diff['mode'] = 'stage3_full_layer1_layer2_rebuild'
            cand_diff['preserves_previous_stage3_state'] = False
            cand_diff['fresh_rebuild'] = True
            return cand_nodes, cand_H, cand_diff, float(cand_r2)

        def validate_fn(nodes: List[_NodeV3],
                        H: torch.Tensor,
                        tag: str) -> Tuple[bool, str]:
            return self._validate_stage3_candidate(
                nodes,
                H,
                S_target=S_target,
                E_target=E_target,
                tag=tag,
            )

        def context_snapshot_fn() -> Dict[str, Any]:
            return copy.deepcopy(
                getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {}
            )

        def context_restore_fn(snapshot: Dict[str, Any]) -> None:
            self._set_fixed_partition_meta(copy.deepcopy(snapshot or {}))

        workflow_result = self.layer4_adjuster.run_block_c_resource_adjustment_loop(
            H=H.detach().clone(),
            nodes=copy.deepcopy(nodes),
            E_target=E_target,
            S_target=S_target,
            rebuild_fn=rebuild_fn,
            validate_fn=validate_fn,
            context_snapshot_fn=context_snapshot_fn,
            context_restore_fn=context_restore_fn,
            output_dir=output_dir,
            flex_max_primary_adjustments=int(flex_max_primary_adjustments),
            external_max_primary_adjustments=int(external_max_primary_adjustments),
            internal_max_primary_adjustments=int(internal_max_primary_adjustments),
            max_h_compensation_ops_per_adjustment=int(max_h_compensation_ops_per_adjustment),
            residual_flex_preferred_23=int(residual_flex_preferred_23),
            residual_flex_min_23=int(residual_flex_min_23),
            residual_side_preferred_23=int(residual_side_preferred_23),
            residual_side_min_23=int(residual_side_min_23),
            h_tolerance=float(h_tolerance),
            outer_cycle=int(outer_idx) + 1,
            # Stage3 cycle numbers are local to this Outer Cycle.  Keep the
            # legacy argument in the helper signature, but never propagate a
            # cross-Outer-Cycle offset into the workflow.
            iteration_offset=0,
            difference_spectrum=dict(current_diff or {}),
            enable_post_external_spectrum=bool(enable_post_external_spectrum),
            spectrum_active_through_cycle=int(spectrum_active_through_cycle),
            spectrum_candidate_limit=int(spectrum_candidate_limit),
            spectrum_max_candidate_trials=int(spectrum_max_candidate_trials),
            spectrum_max_primary_adjustments=int(spectrum_max_primary_adjustments),
            spectrum_min_r2_gain=float(spectrum_min_r2_gain),
            verbose=bool(verbose),
        )

        meta = workflow_result.to_dict()
        meta['stage'] = 'topology_resource_adjustment'
        meta['last_r2'] = float(dict(getattr(workflow_result, 'diff', {}) or {}).get('r2', float('nan')))
        workflow_diagnostics = dict(
            getattr(workflow_result, 'diagnostics', {}) or {}
        )
        raw_completion = dict(
            workflow_diagnostics.get('completion_snapshot', {}) or {}
        )
        raw_terminal = dict(
            workflow_diagnostics.get('terminal_snapshot', {}) or {}
        )
        meta['completion_diagnostics'] = {}
        meta['terminal_diagnostics'] = {}
        if raw_completion.get('stage3_complete', False):
            ledger = dict(
                raw_completion.get('iteration_topology_ledger', {}) or {}
            )
            completion_resource_ledger = dict(
                raw_completion.get('resource_ledger', {})
                or dict(raw_completion.get('external_assembly', {}) or {}).get(
                    'resource_ledger', {}
                )
                or {}
            )
            ledger_for_ownership = {
                **ledger,
                'released_su24_ids': list(
                    completion_resource_ledger.get('released_su24_ids', []) or []
                ),
                'released_resource_ids_by_su': dict(
                    completion_resource_ledger.get(
                        'released_resource_ids_by_su', {}
                    ) or {}
                ),
                'recycled_su24_to_23_ids': list(
                    completion_resource_ledger.get(
                        'recycled_su24_to_23_ids', []
                    ) or []
                ),
            }
            generation_id = int(raw_completion.get('generation_id', 0) or 0)
            raw_products = list(
                raw_completion.get('products', [])
                or ledger.get('completed_products', [])
                or []
            )
            meta['completion_diagnostics'] = {
                'stage3_complete': True,
                'generation_id': generation_id,
                'generation_count': int(raw_completion.get('generation_count', 1) or 1),
                'internal_physical_conversion_count': int(
                    raw_completion.get('internal_physical_conversion_count', 0) or 0
                ),
                'ordinary_internal_primary_adjustment_count': int(
                    raw_completion.get(
                        'ordinary_internal_primary_adjustment_count', 0
                    )
                    or 0
                ),
                'residual_1122_reclassification_primary_adjustment_count': int(
                    raw_completion.get(
                        'residual_1122_reclassification_primary_adjustment_count',
                        0,
                    )
                    or 0
                ),
                'internal_ring_count': int(
                    raw_completion.get('internal_ring_count', 0) or 0
                ),
                'remaining_resource_counts': dict(
                    raw_completion.get('remaining_resource_counts', {}) or {}
                ),
                'generation_metrics': [
                    {
                        'generation_id': int(dict(item or {}).get('generation_id', 0) or 0),
                        'internal_physical_conversion_count': int(
                            dict(item or {}).get(
                                'internal_physical_conversion_count', 0
                            )
                            or 0
                        ),
                        'ordinary_internal_primary_adjustment_count': int(
                            dict(item or {}).get(
                                'ordinary_internal_primary_adjustment_count', 0
                            )
                            or 0
                        ),
                        'residual_1122_reclassification_primary_adjustment_count': int(
                            dict(item or {}).get(
                                'residual_1122_reclassification_primary_adjustment_count',
                                0,
                            )
                            or 0
                        ),
                        'internal_ring_count': int(
                            dict(item or {}).get('internal_ring_count', 0) or 0
                        ),
                    }
                    for item in list(raw_completion.get('generation_metrics', []) or [])
                ],
                'products': [
                    normalize_stage3_product(
                        raw,
                        generation_id=generation_id,
                        index=index,
                    ).to_dict()
                    for index, raw in enumerate(raw_products)
                ],
                'pending_contracts': [
                    dict(item or {})
                    for item in list(raw_completion.get('pending_contracts', []) or [])
                    if dict(item or {})
                ],
                'ownership_ledger': self._stage3_completion_ownership(
                    ledger_for_ownership
                ),
                'node_state_signature': self._stage3_node_state_signature(
                    list(getattr(workflow_result, 'nodes', []) or [])
                ),
                'phase_result': dict(raw_completion.get('phase_result', {}) or {}),
                'classification': dict(raw_completion.get('classification', {}) or {}),
                'external_assembly': dict(raw_completion.get('external_assembly', {}) or {}),
                'resource_ledger': dict(
                    raw_completion.get('resource_ledger', {})
                    or dict(raw_completion.get('external_assembly', {}) or {}).get(
                        'resource_ledger', {}
                    )
                    or {}
                ),
                'resource_accounting': dict(raw_completion.get('resource_accounting', {}) or {}),
            }
        elif raw_terminal:
            terminal_ledger = dict(
                raw_terminal.get('iteration_topology_ledger', {}) or {}
            )
            terminal_resource_ledger = dict(
                raw_terminal.get('resource_ledger', {})
                or dict(raw_terminal.get('external_assembly', {}) or {}).get(
                    'resource_ledger', {}
                )
                or {}
            )
            terminal_ledger_for_ownership = {
                **terminal_ledger,
                'released_su24_ids': list(
                    terminal_resource_ledger.get('released_su24_ids', []) or []
                ),
                'released_resource_ids_by_su': dict(
                    terminal_resource_ledger.get(
                        'released_resource_ids_by_su', {}
                    ) or {}
                ),
                'recycled_su24_to_23_ids': list(
                    terminal_resource_ledger.get(
                        'recycled_su24_to_23_ids', []
                    ) or []
                ),
            }
            terminal_generation_id = int(
                raw_terminal.get('generation_id', 0) or 0
            )
            meta['terminal_diagnostics'] = {
                'stop_reason': str(raw_terminal.get('stop_reason', '') or ''),
                'phase': str(raw_terminal.get('phase', '') or 'unknown'),
                'generation_id': terminal_generation_id,
                'infeasibility_proof': dict(
                    raw_terminal.get('infeasibility_proof', {}) or {}
                ),
                'products': [
                    normalize_stage3_product(
                        raw,
                        generation_id=terminal_generation_id,
                        index=index,
                    ).to_dict()
                    for index, raw in enumerate(
                        list(raw_terminal.get('products', []) or [])
                    )
                ],
                'pending_contracts': [
                    dict(item or {})
                    for item in list(
                        raw_terminal.get('pending_contracts', []) or []
                    )
                ],
                'ownership_ledger': self._stage3_completion_ownership(
                    terminal_ledger_for_ownership
                ),
                'classification': dict(
                    raw_terminal.get('classification', {}) or {}
                ),
                'external_assembly': dict(
                    raw_terminal.get('external_assembly', {}) or {}
                ),
                'resource_ledger': dict(
                    raw_terminal.get('resource_ledger', {})
                    or dict(
                        raw_terminal.get('external_assembly', {}) or {}
                    ).get('resource_ledger', {})
                    or {}
                ),
                'iteration_topology_ledger': terminal_ledger,
                'plan': dict(raw_terminal.get('plan', {}) or {}),
                'node_ids': sorted({
                    int(value)
                    for value in list(raw_terminal.get('node_ids', []) or [])
                }),
                'node_state_signature': self._stage3_node_state_signature(
                    list(getattr(workflow_result, 'nodes', []) or [])
                ),
            }
        completion = dict(meta.get('completion_diagnostics', {}) or {})
        if bool(completion.get('stage3_complete', False)):
            # Persist the exact committed Stage3 ledger for the final report.
            # Layer3 may change latent/template fields, but it must not cause
            # topology products or SU24 ownership to be re-solved.
            resource_ledger = dict(
                completion.get('resource_ledger', {})
                or dict(completion.get('external_assembly', {}) or {}).get(
                    'resource_ledger', {}
                )
                or {}
            )
            completion_ownership = dict(completion.get('ownership_ledger', {}) or {})
            resource_ledger = self._stage3_completion_resource_ledger(
                list(completion.get('products', []) or []),
                completion_ownership,
                resource_ledger,
            )
            completion_snapshot = {
                **completion,
                'node_ids': sorted({
                    int(getattr(node, 'global_id', -1))
                    for node in list(getattr(workflow_result, 'nodes', []) or [])
                }),
                'classification': dict(completion.get('classification', {}) or {}),
                'external_assembly': dict(completion.get('external_assembly', {}) or {}),
                'resource_ledger': resource_ledger,
                'resource_accounting': dict(completion.get('resource_accounting', {}) or {}),
            }
            self._final_stage3_completion_diagnostics = completion_snapshot
            self._final_stage3_snapshot_node_ids = list(completion_snapshot['node_ids'])
            self._final_stage3_snapshot_authoritative = True
            self._final_stage3_terminal_diagnostics = {}
        else:
            self._final_stage3_completion_diagnostics = {}
            self._final_stage3_snapshot_node_ids = []
            self._final_stage3_snapshot_authoritative = False
            self._final_stage3_terminal_diagnostics = dict(meta.get('terminal_diagnostics', {}) or {})
        return (
            list(getattr(workflow_result, 'nodes', []) or []),
            torch.clamp(getattr(workflow_result, 'H', H).detach().clone(), min=0).long(),
            dict(getattr(workflow_result, 'diff', {}) or {}),
            meta,
        )

    def _flatten_stage3_resource_cycles(self,
                                        stage3_meta: Dict[str, Any],
                                        outer_idx: int) -> List[Dict[str, Any]]:
        meta = dict(stage3_meta or {})
        cycles = list(meta.get('cycles', []) or [])
        if cycles:
            return [
                {
                    'outer_cycle': int(outer_idx) + 1,
                    'cycle': int(dict(item or {}).get('cycle', 0) or 0),
                    'phase': str(dict(item or {}).get('phase', '') or ''),
                    'applied': bool(dict(item or {}).get('applied', False)),
                    'stage3_stop_reason': str(
                        dict(item or {}).get('stop_reason', '')
                        or meta.get('stop_reason', '')
                    ),
                    'accepted_cycles_total': int(
                        meta.get('accepted_cycles', 0) or 0
                    ),
                    'accepted_state_retained': bool(
                        dict(item or {}).get('accepted_state_retained', False)
                    ),
                    'rebuild_valid': bool(
                        dict(item or {}).get('rebuild_valid', False)
                    ),
                    'blocked_reason': str(
                        dict(item or {}).get('blocked_reason', '') or ''
                    ),
                    'accepted': bool(
                        dict(item or {}).get('accepted', False)
                    ),
                    'acceptance_reason': str(
                        dict(item or {}).get('acceptance_reason', '') or ''
                    ),
                    'layer2_r2': dict(item or {}).get('layer2_r2', ''),
                    'primary_adjustment_count': int(
                        dict(item or {}).get('primary_adjustment_count', 0) or 0
                    ),
                    'ordinary_internal_primary_adjustment_count': int(
                        dict(item or {}).get(
                            'ordinary_internal_primary_adjustment_count', 0
                        ) or 0
                    ),
                    'residual_1122_reclassification_primary_adjustment_count': int(
                        dict(item or {}).get(
                            'residual_1122_reclassification_primary_adjustment_count',
                            0,
                        ) or 0
                    ),
                    'spectrum_trial_count': int(
                        dict(item or {}).get('spectrum_trial_count', 0) or 0
                    ),
                    'spectrum_max_candidate_trials': int(
                        dict(item or {}).get('spectrum_max_candidate_trials', 0) or 0
                    ),
                    'spectrum_candidate_batches': int(
                        dict(item or {}).get('spectrum_candidate_batches', 0) or 0
                    ),
                    'spectrum_cap_reached': bool(
                        dict(item or {}).get('spectrum_cap_reached', False)
                    ),
                    'spectrum_stop_reason': str(
                        dict(item or {}).get('spectrum_stop_reason', '') or ''
                    ),
                    'physical_conversion_count': int(
                        dict(item or {}).get('physical_conversion_count', 0) or 0
                    ),
                    'h_compensation_count': int(
                        dict(item or {}).get('h_compensation_count', 0) or 0
                    ),
                    'parity_balance_count': int(
                        dict(item or {}).get('parity_balance_count', 0) or 0
                    ),
                    'fresh_rebuild': bool(
                        dict(item or {}).get('fresh_rebuild', False)
                    ),
                }
                for item in cycles
            ]
        return [{
                'outer_cycle': int(outer_idx) + 1,
                'cycle': 0,
                'phase': '',
                'applied': False,
                'stage3_stop_reason': str(meta.get('stop_reason', '')),
                'accepted_cycles_total': int(meta.get('accepted_cycles', 0) or 0),
                'accepted_state_retained': False,
                'rebuild_valid': False,
                'blocked_reason': '',
                'accepted': False,
                'acceptance_reason': '',
                'layer2_r2': '',
                'primary_adjustment_count': 0,
                'ordinary_internal_primary_adjustment_count': 0,
                'residual_1122_reclassification_primary_adjustment_count': 0,
                'spectrum_trial_count': 0,
                'spectrum_max_candidate_trials': 0,
                'spectrum_candidate_batches': 0,
                'spectrum_cap_reached': False,
                'spectrum_stop_reason': '',
                'physical_conversion_count': 0,
                'h_compensation_count': 0,
                'parity_balance_count': 0,
                'fresh_rebuild': False,
            }]

    def _save_final_stage3_resource_summary(self,
                                            rows: Sequence[Dict[str, Any]],
                                            output_dir: str,
                                            completion: Optional[Dict[str, Any]] = None) -> None:
        final_dir = Path(output_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        debug_dir = final_dir / 'debug'
        debug_dir.mkdir(parents=True, exist_ok=True)
        row_list = [dict(row or {}) for row in list(rows or [])]
        completion_snapshot = dict(completion or {})

        cycle_columns = [
            'outer_cycle', 'cycle', 'phase', 'applied', 'stage3_stop_reason',
            'accepted_cycles_total', 'accepted_state_retained', 'rebuild_valid',
            'blocked_reason', 'accepted', 'acceptance_reason', 'layer2_r2',
            'primary_adjustment_count',
            'ordinary_internal_primary_adjustment_count',
            'residual_1122_reclassification_primary_adjustment_count',
            'spectrum_trial_count', 'spectrum_max_candidate_trials',
            'spectrum_candidate_batches', 'spectrum_cap_reached',
            'spectrum_stop_reason',
            'physical_conversion_count', 'h_compensation_count',
            'parity_balance_count', 'fresh_rebuild',
        ]
        compact_cycle_rows = [
            {key: row.get(key, '') for key in cycle_columns}
            for row in row_list
        ]
        pd.DataFrame(compact_cycle_rows, columns=cycle_columns).to_csv(
            final_dir / 'final_stage3_resource_cycles.csv',
            index=False,
        )
        # These files used to contain recursively serialized plans and every
        # transaction from every rebuild.  Remove stale copies when reusing an
        # output directory; final ring/product audits are written separately.
        for stale_path in (
            debug_dir / 'final_stage3_resource_cycle_details.jsonl',
            debug_dir / 'final_stage3_topology_transaction_details.jsonl',
            final_dir / 'final_stage3_topology_transactions.csv',
        ):
            stale_path.unlink(missing_ok=True)
        summary = {
            'cycle_rows': int(len(row_list)),
            'applied_cycle_count': int(sum(1 for row in row_list if bool(row.get('applied', False)))),
            'accepted_cycle_count': int(sum(
                1 for row in row_list
                if str(row.get('stage3_stop_reason', '')) == 'accepted'
                and bool(row.get('rebuild_valid', False))
            )),
            'history_retention': 'latest_outer_cycle_compact_rows_only',
            'outer_cycles_with_stage3': sorted({
                int(row.get('outer_cycle', 0) or 0)
                for row in row_list
                if int(row.get('outer_cycle', 0) or 0) > 0
            }),
            'last_stop_reason': str(row_list[-1].get('stage3_stop_reason', '')) if row_list else '',
            'accepted_spectrum_cycle_count': int(sum(
                1 for row in row_list
                if str(row.get('phase', '')) == 'remaining_pool_spectrum_rebalance'
                and str(row.get('stage3_stop_reason', '')) == 'accepted'
            )),
            'generation_count': int(
                completion_snapshot.get('generation_count', 0) or 0
            ),
            'final_generation_internal_physical_conversion_count': int(
                completion_snapshot.get(
                    'internal_physical_conversion_count', 0
                )
                or 0
            ),
            'final_generation_ordinary_internal_primary_adjustment_count': int(
                completion_snapshot.get(
                    'ordinary_internal_primary_adjustment_count', 0
                )
                or 0
            ),
            'final_generation_residual_1122_reclassification_primary_adjustment_count': int(
                completion_snapshot.get(
                    'residual_1122_reclassification_primary_adjustment_count', 0
                )
                or 0
            ),
            'final_generation_internal_ring_count': int(
                completion_snapshot.get('internal_ring_count', 0) or 0
            ),
            'remaining_resource_counts': {
                str(key): int(value)
                for key, value in dict(
                    completion_snapshot.get('remaining_resource_counts', {}) or {}
                ).items()
            },
            'generation_metrics': [
                {
                    'generation_id': int(dict(item or {}).get('generation_id', 0) or 0),
                    'internal_physical_conversion_count': int(
                        dict(item or {}).get(
                            'internal_physical_conversion_count', 0
                        )
                        or 0
                    ),
                    'ordinary_internal_primary_adjustment_count': int(
                        dict(item or {}).get(
                            'ordinary_internal_primary_adjustment_count', 0
                        )
                        or 0
                    ),
                    'residual_1122_reclassification_primary_adjustment_count': int(
                        dict(item or {}).get(
                            'residual_1122_reclassification_primary_adjustment_count',
                            0,
                        )
                        or 0
                    ),
                    'internal_ring_count': int(
                        dict(item or {}).get('internal_ring_count', 0) or 0
                    ),
                }
                for item in list(
                    completion_snapshot.get('generation_metrics', []) or []
                )
            ],
        }
        with open(final_dir / 'final_stage3_resource_summary.json', 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'summary': summary,
                    'cycles': compact_cycle_rows,
                },
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
    
    def infer(self, S_target: torch.Tensor, E_target: torch.Tensor,
              save_intermediates: bool = True,
              output_dir: str = 'inverse_result',
              eval_lib_path: Optional[str] = None,
              layer1_config: Optional[Dict[str, Any]] = None,
              layer3_config: Optional[Dict[str, Any]] = None,
              stage_configs: Optional[Dict[str, Dict]] = None,
              outer_max_cycles: int = 8,
              outer_improve_eps: float = 1e-4) -> Tuple[List[_NodeV3], torch.Tensor]:
        """
        逆向推理主入口
        
        流程：
        1. Layer0 初始化 SU 直方图
        2. Layer1 -> Layer2 得到并保存完整基线状态
        3. 进入外循环：
           - block_a / block_b:
             作为候选 H 调整，只在重跑 Layer1/2 后通过结构校验且
             Layer2 R2 超过当前接受参考时才采纳
           - block_c_resource:
             每轮执行“分类 -> 组装 -> 精确需求 -> 合法转化 -> 完全重建 Layer1/2”
             的 Stage3 组装循环；资源差谱候选仅以 Layer2-to-Layer2 R2 验收
           - Stage3（包括SU24阶段）完成后执行固定模板 Layer3 z 细化
        4. 返回当前外循环结束时的状态
        
        Args:
            stage_configs: Layer4 正式阶段配置，推荐格式如:
                {
                    'block_a': {...},
                    'block_b': {...},
                    'block_c_resource': {
                        'enabled': True,
                        'flex_max_primary_adjustments': 10,
                        'external_max_primary_adjustments': 10,
                        'internal_max_primary_adjustments': 30,
                    },
                }
            layer1_config 可调参数:
                - neg_peak_threshold, pos_peak_threshold
                - max_candidate_trials_per_region, max_moves_per_region
                - region_adjustment_cycles
                - layer1_multistart_count, layer1_multistart_refine_count
                - layer1_multistart_branch_limit
                Layer2 keeps center/1-hop exact, selects the nearest available
                2-hop tier, and always reconstructs with template center_mu.
            layer3_config 可调参数:
                - max_iters, pos_search_window, neg_assign_window
                - top_k_samples, neg_top_k_peaks, neg_peak_min_sep_ppm
                Layer3 keeps every Layer2 template fixed and always decodes
                candidate z values with G2S. layer3_verbose controls logging.
            outer_max_cycles:
                - Layer4 -> Layer1/2 -> Layer3 宏循环次数上限
            outer_improve_eps:
                - block_a / block_b 的最小 Layer2 R2 提升阈值
        """
        print("=" * 80)
        print("InversePipelineV3: 开始逆向推理")
        print("=" * 80)
        self._final_stage3_completion_diagnostics = {}
        self._final_stage3_terminal_diagnostics = {}
        self._final_stage3_snapshot_node_ids = []
        self._final_stage3_snapshot_authoritative = False
        self.layer4_adjuster._spectrum_search_memory = {}

        # Select the response once from the unmodified target spectrum.  Every
        # subsequent structure reconstruction uses this scalar, while the
        # target tensor itself remains unchanged.
        self._set_aromatic_reconstruction_scale(S_target)
        print(
            f"[Aromatic reconstruction response] fraction="
            f"{self.aromatic_area_fraction:.4f} "
            f"scale={self.aromatic_response_scale:.2f}",
            flush=True,
        )

        layer1_runtime_cfg, layer3_runtime_cfg = self._build_layer_runtime_configs(
            layer1_config=layer1_config,
            layer3_config=layer3_config,
        )
        eval_lib_path = str(eval_lib_path or self.default_template_lib_path or '')
        if not eval_lib_path:
            raise ValueError('infer requires an active template library')
        save_round_outputs = bool(save_intermediates)
        # A run must never inherit a prior run's final CSVs. Archive an
        # existing publication directory before any new intermediate output is
        # written; failed runs therefore cannot masquerade as the previous
        # successful result.
        run_id = str(
            os.environ.get('GRAPHNMR_RUN_ID', '')
            or time.strftime('%Y%m%d-%H%M%S') + f'-{int(os.getpid())}'
        )
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        previous_final = output_root / 'final_outputs'
        if previous_final.exists():
            archive_dir = output_root / 'archive'
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(previous_final), str(archive_dir / f'final_outputs-{run_id}'))
        self._archive_stage3_snapshot_path(
            output_root / 'stage3_input_snapshot',
            run_id=run_id,
        )

        default_stage_configs = {
            'block_a': {
                'max_cycles': 1,
                'max_moves': 6,
                'carbonyl_max_moves': 4,
                'score_rel_threshold': 0.02,
                'peak_rel_threshold': 0.01,
            },
            'block_b': {
                'max_cycles': 1,
                'max_moves_each': 4,
                'peak_rel_threshold': 0.01,
                'verbose': False,
            },
            'block_c_resource': {
                'enabled': True,
                'flex_max_primary_adjustments': 10,
                'external_max_primary_adjustments': 10,
                'internal_max_primary_adjustments': 30,
                'max_h_compensation_ops_per_adjustment': 8,
                'residual_flex_preferred_23': 3,
                'residual_flex_min_23': 2,
                'residual_side_preferred_23': 2,
                'residual_side_min_23': 1,
                'h_tolerance': 0.08,
                'enable_post_external_spectrum': True,
                'spectrum_active_through_cycle': 20,
                'spectrum_candidate_limit': 8,
                'spectrum_max_candidate_trials': 4,
                'spectrum_max_primary_adjustments': 10,
                'spectrum_min_r2_gain': 0.0005,
                'verbose': False,
            },
        }
        supported_block_a_keys = set(default_stage_configs['block_a'])
        supported_block_b_keys = set(default_stage_configs['block_b'])
        if stage_configs is not None:
            for stage, cfg in stage_configs.items():
                if stage in default_stage_configs:
                    default_stage_configs[stage].update(cfg)
                else:
                    raise ValueError(f"未知 stage_configs 键: {stage}")
        block_a_user_cfg = dict((stage_configs or {}).get('block_a', {}) or {})
        unknown_block_a_keys = set(block_a_user_cfg) - supported_block_a_keys
        if unknown_block_a_keys:
            raise ValueError(
                "block_a contains unknown keys: "
                + ", ".join(sorted(str(key) for key in unknown_block_a_keys))
            )
        block_b_user_cfg = dict((stage_configs or {}).get('block_b', {}) or {})
        unknown_block_b_keys = set(block_b_user_cfg) - supported_block_b_keys
        if unknown_block_b_keys:
            raise ValueError(
                "block_b contains unknown keys: "
                + ", ".join(sorted(str(key) for key in unknown_block_b_keys))
            )
        resource_cfg = dict(default_stage_configs.get('block_c_resource', {}) or {})
        resource_enabled = bool(resource_cfg.pop('enabled', True))
        supported_resource_keys = {
            'flex_max_primary_adjustments',
            'external_max_primary_adjustments',
            'internal_max_primary_adjustments',
            'max_h_compensation_ops_per_adjustment',
            'residual_flex_preferred_23',
            'residual_flex_min_23',
            'residual_side_preferred_23',
            'residual_side_min_23',
            'h_tolerance',
            'enable_post_external_spectrum',
            'spectrum_active_through_cycle',
            'spectrum_candidate_limit',
            'spectrum_max_candidate_trials',
            'spectrum_max_primary_adjustments',
            'spectrum_min_r2_gain',
            'verbose',
        }
        unknown_resource_keys = set(resource_cfg) - supported_resource_keys
        if unknown_resource_keys:
            raise ValueError(
                "block_c_resource contains unknown keys: "
                + ", ".join(sorted(str(key) for key in unknown_resource_keys))
            )
        stage3_h_tolerance = float(resource_cfg.get('h_tolerance', 0.08))
        if not 0.0 <= stage3_h_tolerance <= 1.0:
            raise ValueError('block_c_resource.h_tolerance must be between 0 and 1')
        spectrum_active_through_cycle = int(
            resource_cfg.get('spectrum_active_through_cycle', 20)
        )
        if spectrum_active_through_cycle < 0:
            raise ValueError(
                'block_c_resource.spectrum_active_through_cycle must be >= 0'
            )
        resource_cfg['h_tolerance'] = stage3_h_tolerance
        self.layer4_adjuster._h_tolerance = stage3_h_tolerance

        print("\n>>> Layer0: 估计SU直方图")
        H_init = self.estimate_su_histogram(S_target, E_target)
        try:
            self._set_fixed_partition_meta(getattr(self.layer0_estimator, 'fixed_partition_meta', {}) or {})
        except Exception:
            pass
        self._print_histogram_and_elements(H_init, E_target, label='Layer0 Summary')

        print("\n>>> 初始化 Layer1 -> Layer2")
        initial_layer12_nodes, initial_layer12_H, initial_layer12_diff, initial_layer2_r2 = self._run_layer12_cycle(
            H_seed=H_init.detach().clone(),
            S_target=S_target,
            E_target=E_target,
            eval_lib_path=eval_lib_path,
            layer1_runtime_cfg=layer1_runtime_cfg,
            eval_nmr=True,
            eval_output_dir=(str(Path(output_dir) / 'initial_layer12') if save_intermediates and output_dir else None),
            layer2_output_dir=(str(Path(output_dir) / 'initial_layer12') if save_intermediates and output_dir else None),
        )
        hop1_adjuster = getattr(self.layer1_assigner, '_hop1_adjuster', None)
        self.layer4_adjuster.set_spectrum_hop1_template_options(
            getattr(hop1_adjuster, 'shift_options_by_su', {}) or {}
        )
        current_layer2_nodes = copy.deepcopy(initial_layer12_nodes)
        current_layer2_H = initial_layer12_H.detach().clone()
        current_layer2_diff = copy.deepcopy(initial_layer12_diff)
        current_nodes = copy.deepcopy(current_layer2_nodes)
        current_H = current_layer2_H.detach().clone()
        current_diff = copy.deepcopy(current_layer2_diff)
        accepted_layer2_r2 = float(initial_layer2_r2)
        historical_best_layer2_r2 = float(initial_layer2_r2)

        print("\n>>> Outer Loop(Layer4 -> Layer1-2 -> Layer3)")
        stage3_resource_cycle_rows: List[Dict[str, Any]] = []
        stage3_resource_cycle_total = 0
        stage3_resource_accepted_total = 0
        last_layer3_success = False
        outer_cycle_count = max(1, int(outer_max_cycles))
        for outer_idx in range(outer_cycle_count):
            print("\n" + "=" * 80)
            print(f"Outer Cycle {outer_idx + 1}/{outer_cycle_count}")
            print("=" * 80)

            cycle_best_layer2_r2 = float(accepted_layer2_r2)
            cycle_pre_layer3_r2 = float('nan')
            def _candidate_structurally_valid(nodes_local: List[_NodeV3],
                                              H_local: torch.Tensor,
                                              tag: str) -> Tuple[bool, str]:
                meta_before_sync = copy.deepcopy(getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {})
                try:
                    graph_ok, graph_errors = self.layer1_assigner.validate_graph_consistency(
                        nodes=nodes_local,
                        H=H_local,
                        E_target=E_target,
                        verbose=False,
                    )
                except Exception as e:
                    print(f"[{tag}] 图校验失败: {e}")
                    return False, 'graph_check_exception'
                if not bool(graph_ok) or graph_errors:
                    print(f"[{tag}] 回退: 图结构校验未通过，errors={len(graph_errors)}")
                    self.layer1_assigner._print_consistency_errors(graph_errors, prefix=tag, limit=6)
                    return False, 'graph_invalid'
                try:
                    self._sync_fixed_partition_meta_from_nodes(nodes_local)
                except Exception as e:
                    self._set_fixed_partition_meta(meta_before_sync)
                    print(f"[{tag}] 固定锚点元数据同步失败: {e}")
                    return False, 'meta_sync_exception'
                try:
                    hist_diag = self.layer4_adjuster._evaluate_required_hist_constraints(
                        H_local,
                        E_target,
                        S_target=S_target,
                    )
                except Exception as e:
                    self._set_fixed_partition_meta(meta_before_sync)
                    print(f"[{tag}] 直方图校验失败: {e}")
                    return False, 'hist_check_exception'
                if not bool(hist_diag.get('ok', False)):
                    self._set_fixed_partition_meta(meta_before_sync)
                    print(f"[{tag}] 回退: 直方图/固定连接校验未通过，reason={hist_diag.get('reason', 'unknown')}")
                    return False, str(hist_diag.get('reason', 'hist_invalid'))
                return True, 'ok'

            def _candidate_hist_feasible(
                H_local: torch.Tensor,
                tag: str,
                *,
                emit: bool = True,
            ) -> Tuple[bool, str]:
                try:
                    hist_diag = self.layer4_adjuster._evaluate_required_hist_constraints(
                        H_local,
                        E_target,
                        S_target=S_target,
                    )
                except Exception as e:
                    if emit:
                        print(f"[{tag}] 直方图预校验失败: {e}")
                    return False, 'hist_precheck_exception'
                if not bool(hist_diag.get('ok', False)):
                    reason = str(hist_diag.get('reason', 'hist_invalid'))
                    if emit:
                        print(f"[{tag}] 回退: 直方图/固定连接预校验未通过，reason={reason}")
                    return False, reason
                return True, 'ok'

            # Stage A: compare with current best layer2 R2, reject if not improved.
            block_a_cfg = dict(default_stage_configs['block_a'])
            block_a_cycles = max(1, int(block_a_cfg.pop('max_cycles', 1)))
            for block_a_iter in range(block_a_cycles):
                H_block_a_base = current_H.detach().clone()
                nodes_block_a_base = copy.deepcopy(current_nodes)
                diff_block_a_base = copy.deepcopy(current_diff)
                meta_block_a_base = copy.deepcopy(getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {})
                try:
                    H_block_a, moves_a, _meta_a = self.layer4_adjuster.adjust_by_stage(
                        H=H_block_a_base,
                        ppm=diff_block_a_base.get('ppm'),
                        diff=diff_block_a_base.get('diff'),
                        E_target=E_target,
                        S_target=S_target,
                        stage='block_a',
                        nodes=nodes_block_a_base,
                        **block_a_cfg,
                    )
                except Exception as e:
                    try:
                        self._set_fixed_partition_meta(meta_block_a_base)
                    except Exception:
                        pass
                    H_block_a, moves_a = H_block_a_base, []
                    print(f"[Block A #{block_a_iter + 1}] 失败: {e}")

                if not moves_a:
                    try:
                        self._set_fixed_partition_meta(meta_block_a_base)
                    except Exception:
                        pass
                    if int(block_a_iter) == 0:
                        print("[Block A] 无有效调整")
                    break

                hist_ok, _hist_reason = _candidate_hist_feasible(H_block_a, tag=f'BlockA #{block_a_iter + 1}')
                if not bool(hist_ok):
                    try:
                        self._set_fixed_partition_meta(meta_block_a_base)
                    except Exception:
                        pass
                    break

                try:
                    cand_nodes, cand_H, cand_diff, cand_r2 = self._run_layer12_cycle(
                        H_seed=H_block_a.detach().clone(),
                        S_target=S_target,
                        E_target=E_target,
                        eval_lib_path=eval_lib_path,
                        layer1_runtime_cfg=layer1_runtime_cfg,
                        # After Layer4 changes the histogram, rebuild 1-hop
                        # from the current constraints.
                        eval_nmr=False,
                    )
                except Exception as e:
                    try:
                        self._set_fixed_partition_meta(meta_block_a_base)
                    except Exception:
                        pass
                    print(f"[Block A #{block_a_iter + 1}] 回退: layer1/2 failed ({e})")
                    break
                cand_valid, cand_reason = _candidate_structurally_valid(cand_nodes, cand_H, tag='BlockA')
                if cand_valid and float(cand_r2) > float(cycle_best_layer2_r2) + float(outer_improve_eps):
                    current_nodes = cand_nodes
                    current_H = cand_H.detach().clone()
                    current_diff = copy.deepcopy(cand_diff)
                    current_layer2_nodes = copy.deepcopy(cand_nodes)
                    current_layer2_H = cand_H.detach().clone()
                    current_layer2_diff = copy.deepcopy(cand_diff)
                    cycle_pre_layer3_r2 = float(cand_r2)
                    accepted_layer2_r2 = float(cand_r2)
                    historical_best_layer2_r2 = max(float(historical_best_layer2_r2), float(cand_r2))
                    cycle_best_layer2_r2 = float(cand_r2)
                    print(f"[Block A #{block_a_iter + 1}] 接受: layer2_r2={cand_r2:.4f}")
                    self._print_histogram_and_elements(current_H, E_target, label=f'Block A #{block_a_iter + 1} Accepted', nodes=current_nodes)
                    self._print_effective_resource_summary(current_H, label=f'Block A #{block_a_iter + 1} Accepted')
                    continue

                if not bool(cand_valid):
                    try:
                        self._set_fixed_partition_meta(meta_block_a_base)
                    except Exception:
                        pass
                    print(f"[Block A #{block_a_iter + 1}] 回退: candidate invalid ({cand_reason})")
                    break
                try:
                    self._set_fixed_partition_meta(meta_block_a_base)
                except Exception:
                    pass
                print(
                    f"[Block A #{block_a_iter + 1}] 回退: candidate_r2={cand_r2:.4f} "
                    f"<= accept_ref_layer2_r2={cycle_best_layer2_r2:.4f}"
                )
                break

            # Stage B: evaluate each hetero-anchor substage immediately.  The
            # following substage must see the accepted graph/diff from the
            # previous one; otherwise fixed-anchor slots and target demands drift.
            block_b_cfg = dict(default_stage_configs['block_b'])
            block_b_cycles = max(1, int(block_b_cfg.pop('max_cycles', 1)))
            block_b_peak_rel_threshold = float(block_b_cfg.pop('peak_rel_threshold', 0.01))
            block_b_verbose = bool(block_b_cfg.get('verbose', False))
            block_b_families = [
                ('ether', ['ether_2829', 'ether', 'ether_count19', 'ether_mode19']),
                ('amine', ['amine', 'amine_count20', 'amine_mode20']),
                ('thioether', ['thioether', 'thio_count19', 'thio_mode19']),
                # SU21 can only attach to one terminal SU32, so there is no
                # single/double count stage for halogenated aliphatic carbon.
                ('halogen', ['halogen', 'halogen_mode21']),
            ]
            for block_b_iter in range(block_b_cycles):
                block_b_progress = False
                block_b_attempted = False
                for family_idx, (family_name, family_substages) in enumerate(block_b_families):
                    for substage_idx, substage in enumerate(family_substages):
                        nodes_stage_base = copy.deepcopy(current_nodes)
                        H_stage_base = current_H.detach().clone()
                        diff_stage_base = copy.deepcopy(current_diff)
                        meta_stage_base = copy.deepcopy(getattr(self.layer4_adjuster, 'fixed_partition_meta', {}) or {})
                        self.layer4_adjuster.fixed_assignment_precheck = lambda H_candidate: (
                            self.layer1_assigner.precheck_fixed_assignment_feasibility(
                                H_candidate.detach().clone(),
                                E_target,
                            )
                        )
                        try:
                            H_stage, moves_b, _meta_b = self.layer4_adjuster.adjust_by_stage(
                                H=H_stage_base,
                                ppm=diff_stage_base.get('ppm'),
                                diff=diff_stage_base.get('diff'),
                                E_target=E_target,
                                S_target=S_target,
                                stage='block_b',
                                nodes=nodes_stage_base,
                                block_b_substage=str(substage),
                                peak_rel_threshold=block_b_peak_rel_threshold,
                                emit_summary=False,
                                **block_b_cfg,
                            )
                        except Exception as e:
                            try:
                                self._set_fixed_partition_meta(meta_stage_base)
                            except Exception:
                                pass
                            H_stage, moves_b = H_stage_base, []
                            print(
                                f"[Block B Cycle {block_b_iter + 1}] "
                                f"substage={substage} decision=failed reason={e}"
                            )
                            continue

                        if not moves_b and bool(torch.equal(H_stage.detach().cpu(), H_stage_base.detach().cpu())):
                            try:
                                self._set_fixed_partition_meta(meta_stage_base)
                            except Exception:
                                pass
                            continue

                        block_b_attempted = True
                        changes = ", ".join(
                            f"SU{idx}:{before}->{after}"
                            for idx, before, after, _delta in self.layer4_adjuster._summarize_hist_changes(
                                H_stage_base, H_stage
                            )
                        ) or "mode-only"
                        hist_ok, _hist_reason = _candidate_hist_feasible(
                            H_stage,
                            tag=f'BlockB:{family_name}/{substage} #{block_b_iter + 1}',
                            emit=False,
                        )
                        if not bool(hist_ok):
                            try:
                                self._set_fixed_partition_meta(meta_stage_base)
                            except Exception:
                                pass
                            print(
                                f"[Block B Cycle {block_b_iter + 1}] "
                                f"family={family_name} substage={substage}"
                                f"\n  changes: {changes}"
                                f"\n  decision=hist_precheck_rejected reason={_hist_reason}"
                            )
                            continue

                        fixed_precheck_ok, fixed_precheck_reason = self.layer1_assigner.precheck_fixed_assignment_feasibility(
                            H_stage.detach().clone(),
                            E_target,
                        )
                        if not bool(fixed_precheck_ok):
                            try:
                                self._set_fixed_partition_meta(meta_stage_base)
                            except Exception:
                                pass
                            print(
                                f"[Block B Cycle {block_b_iter + 1}] "
                                f"family={family_name} substage={substage} "
                                f"\n  changes: {changes}"
                                f"\n  decision=fixed_precheck_rejected reason={fixed_precheck_reason}"
                            )
                            continue

                        try:
                            layer1_stage_cfg = dict(layer1_runtime_cfg)
                            cand_nodes, cand_H, cand_diff, cand_r2 = self._run_layer12_cycle(
                                H_seed=H_stage.detach().clone(),
                                S_target=S_target,
                                E_target=E_target,
                                eval_lib_path=eval_lib_path,
                                layer1_runtime_cfg=layer1_stage_cfg,
                                eval_nmr=False,
                                emit_summary=False,
                            )
                        except Exception as e:
                            try:
                                self._set_fixed_partition_meta(meta_stage_base)
                            except Exception:
                                pass
                            print(
                                f"[Block B Cycle {block_b_iter + 1}] "
                                f"family={family_name} substage={substage} "
                                f"\n  changes: {changes}"
                                f"\n  decision=layer1_2_failed reason={e}"
                            )
                            continue

                        cand_valid, cand_reason = _candidate_structurally_valid(
                            cand_nodes,
                            cand_H,
                            tag=f'BlockB:{family_name}/{substage}',
                        )
                        if cand_valid and float(cand_r2) > float(cycle_best_layer2_r2) + float(outer_improve_eps):
                            current_nodes = cand_nodes
                            current_H = cand_H.detach().clone()
                            current_diff = copy.deepcopy(cand_diff)
                            current_layer2_nodes = copy.deepcopy(cand_nodes)
                            current_layer2_H = cand_H.detach().clone()
                            current_layer2_diff = copy.deepcopy(cand_diff)
                            cycle_pre_layer3_r2 = float(cand_r2)
                            accepted_layer2_r2 = float(cand_r2)
                            historical_best_layer2_r2 = max(float(historical_best_layer2_r2), float(cand_r2))
                            cycle_best_layer2_r2 = float(cand_r2)
                            block_b_progress = True
                            layer12_summary = dict(
                                cand_diff.get('layer12_summary', {}) or {}
                            )
                            layer2_stats = dict(
                                cand_diff.get('layer2_stats', {}) or {}
                            )
                            print(
                                f"[Block B Cycle {block_b_iter + 1}] "
                                f"family={family_name} substage={substage}\n"
                                f"  changes: {changes}\n"
                                f"  layer1/2: selected={layer12_summary.get('selected_variant_id', 'unknown')} "
                                f"r2={cand_r2:.4f} "
                                f"exact={int(layer2_stats.get('exact_match', 0) or 0)} "
                                f"approx={int(layer2_stats.get('approx_match', 0) or 0)} "
                                f"missing={int(layer2_stats.get('missing', 0) or 0)}\n"
                                f"  decision: accepted reason=layer2_r2_gain"
                            )
                            if block_b_verbose:
                                self._print_histogram_and_elements(
                                    current_H,
                                    E_target,
                                    label=f'Block B:{family_name}/{substage} #{block_b_iter + 1} Accepted',
                                    nodes=current_nodes,
                                )
                                self._print_effective_resource_summary(
                                    current_H,
                                    label=f'Block B:{family_name}/{substage} #{block_b_iter + 1} Accepted',
                                )
                            continue

                        try:
                            self._set_fixed_partition_meta(meta_stage_base)
                        except Exception:
                            pass
                        if not bool(cand_valid):
                            print(
                                f"[Block B Cycle {block_b_iter + 1}] "
                                f"family={family_name} substage={substage} "
                                f"\n  changes: {changes}"
                                f"\n  layer1/2: selected={dict(cand_diff.get('layer12_summary', {}) or {}).get('selected_variant_id', 'unknown')} "
                                f"r2={float(cand_r2):.4f}"
                                f"\n  decision=rejected reason=candidate_invalid:{cand_reason}"
                            )
                            continue
                        print(
                            f"[Block B Cycle {block_b_iter + 1}] "
                            f"family={family_name} substage={substage} "
                            f"\n  changes: {changes}"
                            f"\n  layer1/2: selected={dict(cand_diff.get('layer12_summary', {}) or {}).get('selected_variant_id', 'unknown')} "
                            f"r2={cand_r2:.4f}"
                            f"\n  decision=rejected candidate_r2={cand_r2:.4f} "
                            f"<= accept_ref_layer2_r2={cycle_best_layer2_r2:.4f}"
                        )
                if not bool(block_b_progress) and not bool(block_b_attempted):
                    break

            stage3_complete_for_layer3 = not bool(resource_enabled)
            stage3_nodes = copy.deepcopy(current_layer2_nodes)
            stage3_H = current_layer2_H.detach().clone()
            stage3_diff = copy.deepcopy(current_layer2_diff)
            stage3_meta: Dict[str, Any] = {}
            stage3_stop_reason = (
                'stage3_disabled' if not bool(resource_enabled) else 'stage3_not_started'
            )
            if bool(resource_enabled):
                if int(outer_idx) == 0 and save_round_outputs and output_dir:
                    snapshot_path = self._save_stage3_input_snapshot(
                        output_dir=str(output_dir),
                        nodes=copy.deepcopy(current_layer2_nodes),
                        H=current_layer2_H.detach().clone(),
                        S_target=S_target,
                        E_target=E_target,
                        current_diff=copy.deepcopy(current_layer2_diff),
                        fixed_partition_meta=copy.deepcopy(
                            getattr(
                                self.layer4_adjuster,
                                'fixed_partition_meta',
                                {},
                            )
                            or {}
                        ),
                        layer1_runtime_cfg=copy.deepcopy(layer1_runtime_cfg),
                        stage3_resource_cfg={
                            'enabled': True,
                            **copy.deepcopy(resource_cfg),
                        },
                        eval_lib_path=str(eval_lib_path),
                        outer_idx=int(outer_idx),
                        run_id=str(run_id),
                    )
                    print(
                        f"[Outer Cycle {outer_idx + 1}] "
                        f"Stage3 input snapshot: {snapshot_path}"
                    )
                try:
                    stage3_nodes, stage3_H, stage3_diff, stage3_meta = self._run_topology_stage3_resource_cycles(
                        nodes=copy.deepcopy(current_layer2_nodes),
                        H=current_layer2_H.detach().clone(),
                        current_diff=copy.deepcopy(current_layer2_diff),
                        S_target=S_target,
                        E_target=E_target,
                        eval_lib_path=eval_lib_path,
                        layer1_runtime_cfg=layer1_runtime_cfg,
                        outer_idx=int(outer_idx),
                        flex_max_primary_adjustments=int(resource_cfg.get('flex_max_primary_adjustments', 10)),
                        external_max_primary_adjustments=int(resource_cfg.get('external_max_primary_adjustments', 10)),
                        internal_max_primary_adjustments=int(resource_cfg.get('internal_max_primary_adjustments', 30)),
                        max_h_compensation_ops_per_adjustment=int(resource_cfg.get('max_h_compensation_ops_per_adjustment', 8)),
                        residual_flex_preferred_23=int(resource_cfg.get('residual_flex_preferred_23', 3)),
                        residual_flex_min_23=int(resource_cfg.get('residual_flex_min_23', 2)),
                        residual_side_preferred_23=int(resource_cfg.get('residual_side_preferred_23', 2)),
                        residual_side_min_23=int(resource_cfg.get('residual_side_min_23', 1)),
                        h_tolerance=float(resource_cfg.get('h_tolerance', 0.08)),
                        enable_post_external_spectrum=bool(resource_cfg.get('enable_post_external_spectrum', True)),
                        spectrum_active_through_cycle=spectrum_active_through_cycle,
                        spectrum_candidate_limit=int(resource_cfg.get('spectrum_candidate_limit', 8)),
                        spectrum_max_candidate_trials=int(resource_cfg.get('spectrum_max_candidate_trials', 4)),
                        spectrum_max_primary_adjustments=int(resource_cfg.get('spectrum_max_primary_adjustments', 10)),
                        spectrum_min_r2_gain=float(resource_cfg.get('spectrum_min_r2_gain', 0.0005)),
                        verbose=bool(resource_cfg.get('verbose', False)),
                        output_dir=(
                            str(Path(output_dir) / f'outer_cycle_{outer_idx + 1}' / 'block_c')
                            if save_round_outputs else None
                        ),
                    )
                    latest_stage3_rows = self._flatten_stage3_resource_cycles(
                        stage3_meta,
                        outer_idx=int(outer_idx),
                    )
                    stage3_meta_diagnostics = dict(
                        dict(stage3_meta or {}).get('diagnostics', {}) or {}
                    )
                    stage3_resource_cycle_total += int(
                        stage3_meta_diagnostics.get(
                            'total_cycle_count', len(latest_stage3_rows)
                        )
                        or len(latest_stage3_rows)
                    )
                    stage3_resource_accepted_total += int(
                        dict(stage3_meta or {}).get('accepted_cycles', 0) or 0
                    )
                    # Previous outer-cycle rows describe graphs that no longer
                    # exist.  Keep only the latest bounded summary in memory.
                    stage3_resource_cycle_rows = latest_stage3_rows
                    stage3_stop_reason = str(stage3_meta.get('stop_reason', '') or '')
                    stage3_complete_for_layer3 = stage3_stop_reason == 'stage3_complete'
                    if stage3_complete_for_layer3:
                        current_nodes = copy.deepcopy(stage3_nodes)
                        current_H = stage3_H.detach().clone()
                        current_diff = copy.deepcopy(stage3_diff)
                        current_layer2_nodes = copy.deepcopy(stage3_nodes)
                        current_layer2_H = stage3_H.detach().clone()
                        current_layer2_diff = copy.deepcopy(stage3_diff)
                        stage3_layer2_r2 = float(stage3_diff.get('r2', float('nan')))
                        if np.isfinite(stage3_layer2_r2):
                            accepted_layer2_r2 = float(stage3_layer2_r2)
                            historical_best_layer2_r2 = max(
                                float(historical_best_layer2_r2),
                                float(stage3_layer2_r2),
                            )
                            cycle_pre_layer3_r2 = float(stage3_layer2_r2)
                    status = '完成' if stage3_complete_for_layer3 else '未完成'
                    print(
                        f"[Block C:resource] {status}: "
                        f"accepted_cycles={int(stage3_meta.get('accepted_cycles', 0) or 0)} "
                        f"stop={stage3_stop_reason}"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Block C resource workflow crashed: {e}"
                    ) from e
            else:
                current_nodes = copy.deepcopy(current_layer2_nodes)
                current_H = current_layer2_H.detach().clone()
                current_diff = copy.deepcopy(current_layer2_diff)
                cycle_pre_layer3_r2 = float(current_layer2_diff.get('r2', float('nan')))

            if stage3_complete_for_layer3:
                layer3_input_nodes = copy.deepcopy(current_nodes)
                layer3_input_H = current_H.detach().clone()
                layer3_input_diff = copy.deepcopy(current_diff)
                try:
                    layer3_nodes, layer3_r2 = self.layer3_adjust_templates(
                        nodes=copy.deepcopy(layer3_input_nodes),
                        S_target=S_target,
                        E_target=E_target,
                        lib_path=eval_lib_path,
                        output_dir=None,
                        **layer3_runtime_cfg,
                    )
                except Exception as e:
                    raise RuntimeError(f"Outer Cycle {outer_idx + 1} Layer3 failed: {e}") from e
                layer3_diagnostics = dict(
                    getattr(self.layer3_estimator, 'last_diagnostics', {}) or {}
                )
                layer3_accepted = bool(layer3_diagnostics.get('accepted', True))
                if layer3_accepted:
                    current_nodes = layer3_nodes
                else:
                    current_nodes = layer3_input_nodes
                    current_H = layer3_input_H
                    current_diff = layer3_input_diff
                    print(
                        f"[Outer Cycle {outer_idx + 1} Layer3] rejected: "
                        f"candidate_r2={float(layer3_diagnostics.get('candidate_r2', layer3_r2)):.4f} "
                        f"< baseline_r2={float(layer3_diagnostics.get('baseline_r2', cycle_pre_layer3_r2)):.4f}; "
                        "retained pre-Layer3 state"
                    )
                last_layer3_success = True
            else:
                print(
                    f"[Outer Cycle {outer_idx + 1}] Layer3 skipped: "
                    "Stage3 topology assembly is incomplete"
                )
                last_layer3_success = False
                if save_round_outputs:
                    self._save_outer_cycle_snapshot(
                        outer_idx=int(outer_idx),
                        nodes=copy.deepcopy(stage3_nodes),
                        H=stage3_H.detach().clone(),
                        E_target=E_target,
                        S_target=S_target,
                        output_root=output_root,
                        run_id=run_id,
                        stage3_complete=False,
                        stage3_meta=stage3_meta,
                        selected_diff=copy.deepcopy(stage3_diff),
                        pre_layer3_r2=float(
                            stage3_diff.get('r2', cycle_pre_layer3_r2)
                        ),
                        post_layer3_r2=float('nan'),
                        layer3_committed=False,
                    )
                current_nodes = copy.deepcopy(current_layer2_nodes)
                current_H = current_layer2_H.detach().clone()
                current_diff = copy.deepcopy(current_layer2_diff)
                continue

            if layer3_accepted:
                current_H = self._histogram_from_nodes(current_nodes).detach().clone()
                current_diff = self._compute_difference_spectrum_from_nodes_mu(
                    nodes=current_nodes,
                    S_target=S_target,
                )
                current_diff['source_stage'] = 'post_layer3'
                current_diff['spectral_contract'] = 'template_sample_mu_layer3'
            else:
                current_diff['layer3_candidate_diagnostics'] = layer3_diagnostics
            cycle_post_layer3_r2 = float(current_diff.get('r2', float('nan')))
            cycle_pre_layer3_txt = (
                f"{cycle_pre_layer3_r2:.4f}"
                if np.isfinite(cycle_pre_layer3_r2) else
                "n/a"
            )
            cycle_post_layer3_txt = (
                f"{cycle_post_layer3_r2:.4f}"
                if np.isfinite(cycle_post_layer3_r2) else
                "nan"
            )
            print(
                f"[Outer Cycle {outer_idx + 1}] Summary: "
                f"accept_ref_layer2_r2={accepted_layer2_r2:.4f}, "
                f"historical_best_layer2_r2={historical_best_layer2_r2:.4f}, "
                f"pre_layer3_r2={cycle_pre_layer3_txt}, "
                f"post_layer3_r2={cycle_post_layer3_txt}, "
                f"layer3_committed={bool(layer3_accepted)}"
            )
            self._print_histogram_and_elements(
                current_H,
                E_target,
                label=f'Outer Cycle {outer_idx + 1} Selected State',
                nodes=current_nodes,
            )
            self._print_effective_resource_summary(
                current_H,
                label=f'Outer Cycle {outer_idx + 1} Selected State',
            )

            if save_round_outputs:
                outer_dir = str(Path(output_dir) / f"outer_cycle_{outer_idx + 1}")
                Path(outer_dir).mkdir(parents=True, exist_ok=True)
                try:
                    visualize_su_distribution(
                        current_H.long().detach().cpu(),
                        f'Outer_{outer_idx + 1}',
                        save_dir=outer_dir,
                    )
                except Exception as exc:
                    print(
                        f"[Outer Cycle {outer_idx + 1} Snapshot] "
                        f"SU visualization failed: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                try:
                    self._save_final_nmr_comparison(
                        nodes=current_nodes,
                        S_target=S_target,
                        E_target=E_target,
                        output_dir=str(Path(outer_dir) / 'nmr_comparison'),
                        save_full_outputs=False,
                    )
                except Exception as exc:
                    print(
                        f"[Outer Cycle {outer_idx + 1} Snapshot] "
                        f"NMR comparison failed: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                self._save_outer_cycle_snapshot(
                    outer_idx=int(outer_idx),
                    nodes=copy.deepcopy(current_nodes),
                    H=current_H.detach().clone(),
                    E_target=E_target,
                    S_target=S_target,
                    output_root=output_root,
                    run_id=run_id,
                    stage3_complete=bool(stage3_complete_for_layer3),
                    stage3_meta=stage3_meta,
                    selected_diff=copy.deepcopy(current_diff),
                    pre_layer3_r2=float(cycle_pre_layer3_r2),
                    post_layer3_r2=float(cycle_post_layer3_r2),
                    layer3_committed=bool(layer3_accepted),
                )

        if not bool(last_layer3_success):
            failure_dir = output_root / 'failure_outputs' / run_id
            failure_dir.mkdir(parents=True, exist_ok=True)
            terminal_diagnostics = dict(
                getattr(self, '_final_stage3_terminal_diagnostics', {}) or {}
            )
            failure_payload = {
                'ok': False,
                'run_id': str(run_id),
                'stop_reason': str(stage3_stop_reason or 'stage3_incomplete'),
                'terminal_diagnostics': terminal_diagnostics,
                'stage3_cycles': list(stage3_resource_cycle_rows or []),
                'stage3_cycle_total': int(stage3_resource_cycle_total),
                'stage3_accepted_total': int(stage3_resource_accepted_total),
            }
            status_tmp = failure_dir / '.run_status.json.tmp'
            status_tmp.write_text(
                json.dumps(failure_payload, ensure_ascii=False, indent=2, default=str),
                encoding='utf-8',
            )
            status_tmp.replace(failure_dir / 'run_status.json')
            proof_tmp = failure_dir / '.terminal_proof.json.tmp'
            proof_tmp.write_text(
                json.dumps(
                    dict(terminal_diagnostics.get('infeasibility_proof', {}) or {}),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding='utf-8',
            )
            proof_tmp.replace(failure_dir / 'terminal_proof.json')
            raise RuntimeError('final_result_requires_completed_stage3_and_layer3')

        print(
            "[Final Topology] using final accepted spectral state; no extra Layer1/2 rebuild",
            flush=True,
        )

        nodes = copy.deepcopy(current_nodes)
        H_final = self._histogram_from_nodes(nodes).detach().clone()
        final_consistency = self._audit_final_su_identity(nodes)
        if not bool(final_consistency.get('ok', False)):
            raise RuntimeError(
                'final_su_identity_mismatch:'
                + json.dumps(final_consistency, ensure_ascii=False, sort_keys=True)
            )

        try:
            visualize_su_distribution(H_final.long().detach().cpu(), 'Final', save_dir=output_dir)
        except Exception:
            pass
        final_output_dir = Path(output_dir) / 'final_outputs'
        final_output_tmp = Path(output_dir) / f'.final_outputs-{run_id}.tmp'
        if final_output_tmp.exists():
            shutil.rmtree(final_output_tmp)
        self._save_final_nmr_comparison(
            nodes=nodes,
            S_target=S_target,
            E_target=E_target,
            output_dir=str(final_output_tmp),
        )
        self._save_final_stage3_resource_summary(
            rows=stage3_resource_cycle_rows,
            output_dir=str(final_output_tmp),
            completion=dict(
                getattr(self, '_final_stage3_completion_diagnostics', {}) or {}
            ),
        )
        final_output_tmp.replace(final_output_dir)
        print("\n" + "=" * 80)
        print("最终结果汇总")
        print("=" * 80)
        final_selected_r2 = float(current_diff.get('r2', float('nan')))
        final_selected_txt = (
            f"{final_selected_r2:.4f}"
            if np.isfinite(final_selected_r2) else
            "nan"
        )
        print(
            f"[Final R2] layer2_accept_ref={accepted_layer2_r2:.4f}, "
            f"historical_best_layer2_r2={historical_best_layer2_r2:.4f}, "
            f"final_selected_r2={final_selected_txt}, "
            f"spectral_contract={str(current_diff.get('spectral_contract', 'unknown'))}"
        )
        stage3_last_stop = str(stage3_resource_cycle_rows[-1].get('stage3_stop_reason', '')) if stage3_resource_cycle_rows else ''
        print(
            f"[Final Stage3 Resource] cycles={stage3_resource_cycle_total} "
            f"retained={len(stage3_resource_cycle_rows)} "
            f"accepted={stage3_resource_accepted_total} "
            f"last_stop={stage3_last_stop}"
        )
        self._print_histogram_and_elements(H_final, E_target, label='Final Summary', nodes=nodes)
        self._print_final_topology_specs(
            nodes,
            S_target=S_target,
            E_target=E_target,
        )
        print("\n" + "=" * 80)
        print("逆向推理完成！")
        print("=" * 80)

        return nodes, H_final

    def _compute_difference_spectrum_from_nodes_mu(
        self,
        nodes: List[_NodeV3],
        S_target: torch.Tensor,
    ) -> Dict[str, object]:
        target = S_target.to(self.device).flatten()
        ppm_axis = PPM_AXIS.to(self.device).flatten()
        reconstructed = self.reconstruct_spectrum(nodes).to(self.device).flatten()
        evaluation = evaluate_spectrum_reconstruction(
            target,
            reconstructed,
            ppm_axis=ppm_axis,
        )
        ppm_eval = evaluation.get('ppm_axis', ppm_axis)
        difference = (
            evaluation['S_target'] - evaluation['S_recon']
        ).detach().cpu().numpy()
        return {
            'ppm': ppm_eval.detach().cpu().numpy(),
            'diff': difference,
            'r2': float(evaluation.get('r2', 0.0)),
            'aromatic_area_fraction': float(self.aromatic_area_fraction),
            'aromatic_response_scale': float(self.aromatic_response_scale),
            'aromatic_su_types': [5, 6, 7, 8, 9, 10, 11, 12, 13],
            'target_spectrum_scaled': False,
            'reconstruction_aromatic_peaks_scaled': True,
        }

    def _histogram_from_nodes(self, nodes: List[_NodeV3]) -> torch.Tensor:
        device = self.device
        H = torch.zeros(NUM_SU_TYPES, dtype=torch.long, device=device)
        for n in nodes:
            su = int(n.su_type)
            if 0 <= su < NUM_SU_TYPES:
                H[su] += 1
        return H
    
    def estimate_su_histogram(self, S_target: torch.Tensor, E_target: torch.Tensor) -> torch.Tensor:
        """委托给Layer0Estimator执行SU直方图估计"""
        return self.layer0_estimator.estimate_su_histogram(S_target, E_target)

    def run_layer12(
        self,
        S_target: torch.Tensor,
        E_target: torch.Tensor,
        *,
        H_seed: Optional[torch.Tensor] = None,
        save_intermediates: bool = True,
        output_dir: Optional[str] = None,
        eval_lib_path: Optional[str] = None,
        layer1_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[_NodeV3], torch.Tensor, Dict[str, object], float]:
        """Run Layer0/1/2 once with the fixed inverse spectral contract."""
        started = time.perf_counter()
        target = S_target.to(self.device).flatten()
        self._set_aromatic_reconstruction_scale(target)
        elements = E_target.to(self.device).flatten()
        layer1_runtime_config, _layer3_config = self._build_layer_runtime_configs(
            layer1_config=layer1_config,
        )
        resolved_lib = str(eval_lib_path or self.default_template_lib_path or '')
        if not resolved_lib:
            raise ValueError('run_layer12 requires an active template library')

        layer0_started = time.perf_counter()
        if H_seed is None:
            histogram = self.estimate_su_histogram(target, elements)
            self._set_fixed_partition_meta(
                getattr(self.layer0_estimator, 'fixed_partition_meta', {}) or {}
            )
        else:
            histogram = H_seed.to(self.device).detach().clone()
        layer0_elapsed = time.perf_counter() - layer0_started

        stage_output = None
        if save_intermediates and output_dir:
            stage_output = str(Path(output_dir))
            Path(stage_output).mkdir(parents=True, exist_ok=True)
        cycle_started = time.perf_counter()
        nodes, final_histogram, difference, layer2_r2 = self._run_layer12_cycle(
            H_seed=histogram.detach().clone(),
            S_target=target,
            E_target=elements,
            eval_lib_path=resolved_lib,
            layer1_runtime_cfg=layer1_runtime_config,
            eval_nmr=True,
            eval_output_dir=stage_output,
            layer2_output_dir=stage_output,
        )
        difference['layer0_elapsed_s'] = float(layer0_elapsed)
        difference['layer12_compute_elapsed_s'] = float(
            time.perf_counter() - cycle_started
        )
        difference['run_layer12_total_elapsed_s'] = float(
            time.perf_counter() - started
        )
        return nodes, final_histogram, difference, float(layer2_r2)
    def layer1_assign(
        self,
        H_init: torch.Tensor,
        S_target: torch.Tensor,
        E_target: torch.Tensor,
        eval_nmr: bool = True,
        eval_output_dir: Optional[str] = 'inverse_result',
        eval_lib_path: Optional[str] = None,
        layer1_config: Optional[Dict[str, Any]] = None,
    ) -> List[_NodeV3]:
        # Keep direct Layer1 entry points on the same reconstruction contract
        # as ``infer``: the target is read-only, while aromatic response is
        # applied only to structure-derived spectra.
        self._set_aromatic_reconstruction_scale(S_target)
        runtime_config, _layer3_config = self._build_layer_runtime_configs(
            layer1_config=layer1_config,
        )
        for key in (
            'layer1_multistart_count',
            'layer1_multistart_refine_count',
            'layer1_multistart_branch_limit',
        ):
            runtime_config.pop(key, None)
        return self.layer1_assigner.layer1_assign(
            H_init=H_init,
            S_target=S_target,
            E_target=E_target,
            eval_nmr=eval_nmr,
            eval_output_dir=eval_output_dir,
            eval_lib_path=eval_lib_path or self.default_template_lib_path,
            **runtime_config,
        )

    def evaluate_layer1_nmr_with_library(
        self,
        nodes: List[_NodeV3],
        S_target: torch.Tensor,
        lib_path: Optional[str] = None,
        output_dir: Optional[str] = 'inverse_result',
    ) -> Dict[str, float]:
        return self.layer1_assigner.evaluate_layer1_nmr_with_library(
            nodes=nodes,
            S_target=S_target,
            lib_path=lib_path or self.default_template_lib_path,
            output_dir=output_dir,
        )

    def layer2_assign(self, nodes: List[_NodeV3],
                      S_target: torch.Tensor, E_target: torch.Tensor,
                      lib_path: Optional[str] = None,
                      output_dir: Optional[str] = None,
                      verbose: bool = False,
                      emit_summary: bool = True) -> List[_NodeV3]:
        """
        Layer2: 委托给Layer2Estimator进行2-hop推导和模板检索
        
        功能：
        1. 从1-hop邻居推导2-hop邻居
        2. 模板精确匹配和近似匹配
        3. 保存模板center_z并使用模板center_mu
        4. NMR谱图重建和评估
        """
        self._set_aromatic_reconstruction_scale(S_target)
        lib_path = lib_path or self.default_template_lib_path
        if lib_path:
            try:
                if getattr(self.layer2_estimator, 'lib_path', None) != lib_path:
                    self.layer2_estimator.lib_path = lib_path
                    if hasattr(self.layer2_estimator, '_template_cache'):
                        self.layer2_estimator._template_cache = None
            except Exception:
                pass
        # 委托给Layer2Estimator
        return self.layer2_estimator.layer2_assign(
            nodes=nodes,
            S_target=S_target,
            E_target=E_target,
            output_dir=output_dir,
            verbose=bool(verbose),
            emit_summary=bool(emit_summary),
        )

    def _share_layer2_template_cache_with_layer3(self, lib_path: Optional[str]) -> bool:
        resolved_path = str(lib_path or self.default_template_lib_path or "")
        layer2_path = str(getattr(self.layer2_estimator, "lib_path", "") or "")
        layer2_cache = getattr(self.layer2_estimator, "_template_cache", None)
        if not resolved_path or layer2_path != resolved_path or not isinstance(layer2_cache, dict):
            return False
        self.layer3_estimator.lib_path = resolved_path
        self.layer3_estimator._template_cache = layer2_cache
        return True
    
    # ========================================================================
    # Layer3: z调优（改进版）
    # ========================================================================
    
    def layer3_adjust_templates(self,
                                nodes: List[_NodeV3],
                                S_target: torch.Tensor,
                                E_target: torch.Tensor,
                                max_iters: int = 150,
                                lib_path: Optional[str] = None,
                                output_dir: Optional[str] = None,
                                pos_search_window: float = 15.0,
                                neg_assign_window: float = 2.0,
                                top_k_samples: int = 8,
                                neg_top_k_peaks: int = 10,
                                neg_peak_min_sep_ppm: float = 0.8,
                                layer3_verbose: bool = True) \
                                -> Tuple[List[_NodeV3], float]:
        """
        Layer3: 通过调整z向量优化谱图匹配
        
        改进点：
        1. 分段优先优化：集中火力攻克尖峰区域
        2. 候选池扩展：增加多样性采样
        3. 早停机制：R²达标提前终止
        """
        self._set_aromatic_reconstruction_scale(S_target)
        return self.layer3_adjust_templates_impl(
            nodes=nodes,
            S_target=S_target,
            E_target=E_target,
            max_iters=int(max_iters),
            lib_path=lib_path,
            output_dir=output_dir,
            pos_search_window=float(pos_search_window),
            neg_assign_window=float(neg_assign_window),
            top_k_samples=int(top_k_samples),
            neg_top_k_peaks=int(neg_top_k_peaks),
            neg_peak_min_sep_ppm=float(neg_peak_min_sep_ppm),
            verbose=bool(layer3_verbose),
        )

    def layer3_adjust_templates_impl(
        self,
        nodes: List[_NodeV3],
        S_target: torch.Tensor,
        E_target: torch.Tensor,
        max_iters: int = 150,
        lib_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        pos_search_window: float = 15.0,
        neg_assign_window: float = 2.0,
        top_k_samples: int = 8,
        neg_top_k_peaks: int = 10,
        neg_peak_min_sep_ppm: float = 0.8,
        verbose: bool = True,
    ) -> Tuple[List[_NodeV3], float]:
        lib_path = lib_path or self.default_template_lib_path
        if lib_path:
            try:
                if getattr(self.layer3_estimator, 'lib_path', None) != lib_path:
                    self.layer3_estimator.lib_path = lib_path
                    if hasattr(self.layer3_estimator, '_template_cache'):
                        self.layer3_estimator._template_cache = None
            except Exception:
                pass
        shared_template_cache = self._share_layer2_template_cache_with_layer3(lib_path)
        if verbose:
            print(
                "[Layer3 template library] "
                + ("reusing Layer2 in-memory cache" if shared_template_cache else "loading Layer3 cache")
            )
        return self.layer3_estimator.layer3_adjust_templates(
            nodes=nodes,
            S_target=S_target,
            E_target=E_target,
            max_iters=int(max_iters),
            output_dir=output_dir,
            pos_search_window=float(pos_search_window),
            neg_assign_window=float(neg_assign_window),
            top_k_samples=int(top_k_samples),
            neg_top_k_peaks=int(neg_top_k_peaks),
            neg_peak_min_sep_ppm=float(neg_peak_min_sep_ppm),
            verbose=bool(verbose),
        )

    def _save_final_nmr_comparison(self,
                                   nodes: List[_NodeV3],
                                   S_target: torch.Tensor,
                                   E_target: torch.Tensor,
                                   output_dir: str,
                                   save_full_outputs: bool = True) -> None:
        final_dir = Path(output_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        diff_info = self._compute_difference_spectrum_from_nodes_mu(
            nodes=nodes,
            S_target=S_target,
        )
        ppm_np = np.asarray(diff_info.get('ppm', []), dtype=np.float64)
        S_target_np = np.asarray(S_target.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        S_recon_np = self.reconstruct_spectrum(nodes).detach().cpu().numpy().astype(np.float64).reshape(-1)
        n = min(len(ppm_np), len(S_target_np), len(S_recon_np))
        ppm_np = ppm_np[:n]
        S_target_np = S_target_np[:n]
        S_recon_np = S_recon_np[:n]

        pd.DataFrame({
            'ppm': ppm_np,
            'target': S_target_np,
            'reconstructed': S_recon_np,
        }).to_csv(final_dir / 'final_nmr_comparison.csv', index=False)

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.fill_between(ppm_np, 0, S_target_np, alpha=0.25, color='#1f77b4', label='Target')
            ax.fill_between(ppm_np, 0, S_recon_np, alpha=0.25, color='#ff7f0e', label='Reconstructed')
            ax.plot(ppm_np, S_target_np, lw=1.8, color='#1f77b4', label='Target')
            ax.plot(ppm_np, S_recon_np, lw=1.8, color='#ff7f0e', linestyle='--', label='Reconstructed')
            ax.invert_xaxis()
            ax.set_xlabel('Chemical Shift (ppm)')
            ax.set_ylabel('Intensity')
            ax.set_title('Final Target vs Reconstructed NMR')
            ax.legend(frameon=False)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(final_dir / 'final_nmr_comparison.png', dpi=300)
            plt.close(fig)
        except Exception:
            pass

        if bool(save_full_outputs):
            self._save_final_node_exports(nodes=nodes, output_dir=str(final_dir))
            self._save_final_result_summary(
                nodes=nodes,
                E_target=E_target,
                S_target=S_target,
                output_dir=str(final_dir),
            )

    def _build_final_node_rows(self, nodes: List[_NodeV3]) -> List[Dict[str, Any]]:
        su_names = [name for name, _ in SU_DEFS]
        rows: List[Dict[str, Any]] = []
        for n in nodes:
            su_idx = int(getattr(n, 'su_type', -1))
            hop1_ms = tuple(multiset_from_counter(getattr(n, 'hop1_su', {}) or {}))
            hop2_ms = tuple(multiset_from_counter(getattr(n, 'hop2_su', {}) or {}))
            hop1_ids = [int(x) for x in list(getattr(n, 'hop1_ids', []) or [])]
            fixed_ids = sorted(int(x) for x in set(getattr(n, 'fixed_hop1_ids', set()) or set()))
            score_components = dict(getattr(n, 'score_components', {}) or {})
            z_vec = getattr(n, 'z_vec', None)
            z_norm = None
            z_head4 = None
            if isinstance(z_vec, torch.Tensor) and int(z_vec.numel()) > 0:
                try:
                    z_norm = float(z_vec.detach().float().norm().item())
                    z_head4 = ",".join(
                        f"{float(v):.6f}" for v in z_vec.detach().flatten()[:4].cpu().tolist()
                    )
                except Exception:
                    z_norm = None
                    z_head4 = None
            rows.append({
                'global_id': int(getattr(n, 'global_id', -1)),
                'su_type': int(su_idx),
                'su_name': su_names[su_idx] if 0 <= su_idx < len(su_names) else str(su_idx),
                'special_anchor_partition': str(getattr(n, 'special_anchor_partition', '')) if getattr(n, 'special_anchor_partition', None) is not None else '',
                'special_anchor_mode': str(getattr(n, 'special_anchor_mode', '')) if getattr(n, 'special_anchor_mode', None) is not None else '',
                'target_hop1_degree': int(getattr(n, 'target_hop1_degree', -1)) if getattr(n, 'target_hop1_degree', None) is not None else None,
                'target_fixed_anchor_count': int(getattr(n, 'target_fixed_anchor_count', -1)) if getattr(n, 'target_fixed_anchor_count', None) is not None else None,
                'actual_hop1_degree': int(sum((getattr(n, 'hop1_su', {}) or {}).values())) if getattr(n, 'hop1_su', None) is not None else 0,
                'hop1_ids': str(hop1_ids),
                'hop1_ms': str(list(hop1_ms)),
                'hop2_ms': str(list(hop2_ms)),
                'mu': float(getattr(n, 'mu', 0.0) or 0.0),
                'pi': float(getattr(n, 'pi', 0.0) or 0.0),
                'template_key': str(getattr(n, 'template_key', None)),
            })
        return rows

    def _save_final_node_exports(self, nodes: List[_NodeV3], output_dir: str) -> None:
        consistency = self._audit_final_su_identity(nodes)
        if not bool(consistency.get('ok', False)):
            raise RuntimeError(
                'final_su_identity_mismatch:'
                + json.dumps(consistency, ensure_ascii=False, sort_keys=True)
            )
        final_dir = Path(output_dir)
        final_dir.mkdir(parents=True, exist_ok=True)

        rows = self._build_final_node_rows(nodes)
        node_columns = [
            'global_id', 'su_type', 'su_name',
            'special_anchor_partition', 'special_anchor_mode',
            'target_hop1_degree', 'target_fixed_anchor_count', 'actual_hop1_degree',
            'hop1_ids', 'hop1_ms', 'hop2_ms',
            'mu', 'pi', 'template_key',
        ]
        df_nodes = pd.DataFrame(rows)
        if not df_nodes.empty:
            df_nodes.sort_values(['global_id', 'su_type'], kind='stable').to_csv(
                final_dir / 'final_nodes_by_global_id.csv', index=False
            )
            df_nodes.sort_values(['su_type', 'global_id'], kind='stable').to_csv(
                final_dir / 'final_nodes_by_su_then_global.csv', index=False
            )
        else:
            pd.DataFrame(columns=node_columns).to_csv(final_dir / 'final_nodes_by_global_id.csv', index=False)
            pd.DataFrame(columns=node_columns).to_csv(final_dir / 'final_nodes_by_su_then_global.csv', index=False)

        su_names = [name for name, _ in SU_DEFS]
        counts = [0] * len(su_names)
        for n in nodes:
            try:
                su_idx = int(getattr(n, 'su_type', -1))
            except Exception:
                continue
            if 0 <= su_idx < len(counts):
                counts[su_idx] += 1
        df_dist = pd.DataFrame([
            {
                'su_type': int(idx),
                'su_name': su_names[idx] if 0 <= idx < len(su_names) else str(idx),
                'count': int(cnt),
            }
            for idx, cnt in enumerate(counts)
        ])
        df_dist.sort_values(['su_type'], kind='stable').to_csv(
            final_dir / 'final_su_distribution.csv', index=False
        )

    def _save_su_distribution_only(
        self,
        nodes: List[_NodeV3],
        output_dir: str,
    ) -> None:
        """Write the node-derived SU distribution without topology auditing.

        Outer-cycle snapshots must retain this file even when a topology
        export is unavailable or fails its audit.  Counts intentionally come
        from the same node list used by ``_save_final_node_exports``.
        """
        final_dir = Path(output_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        su_names = [name for name, _ in SU_DEFS]
        counts = [0] * len(su_names)
        for node in list(nodes or []):
            try:
                su_idx = int(getattr(node, 'su_type', -1))
            except Exception:
                continue
            if 0 <= su_idx < len(counts):
                counts[su_idx] += 1
        pd.DataFrame([
            {
                'su_type': int(idx),
                'su_name': su_names[idx] if 0 <= idx < len(su_names) else str(idx),
                'count': int(count),
            }
            for idx, count in enumerate(counts)
        ]).to_csv(final_dir / 'final_su_distribution.csv', index=False)

    def _save_outer_cycle_snapshot(
        self,
        *,
        outer_idx: int,
        nodes: List[_NodeV3],
        H: torch.Tensor,
        E_target: torch.Tensor,
        S_target: Optional[torch.Tensor],
        output_root: Path,
        run_id: str,
        stage3_complete: bool,
        stage3_meta: Optional[Dict[str, Any]],
        selected_diff: Optional[Dict[str, Any]],
        pre_layer3_r2: float,
        post_layer3_r2: float,
        layer3_committed: bool,
    ) -> Dict[str, Any]:
        """Publish an outer-cycle result as one bounded, auditable snapshot.

        Each file is attempted independently so a topology audit cannot hide
        the SU distribution.  The temporary directory is published only
        after the status file has recorded all errors and expected paths.
        """
        cycle_dir = Path(output_root) / f'outer_cycle_{int(outer_idx) + 1}'
        cycle_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = cycle_dir / f'.final_outputs-{run_id}.tmp'
        final_dir = cycle_dir / 'final_outputs'
        errors: List[Dict[str, str]] = []
        result: Dict[str, Any] = {
            'outer_cycle': int(outer_idx) + 1,
            'run_id': str(run_id),
            'complete': bool(stage3_complete),
            'published': False,
            'errors': errors,
        }
        try:
            stale_failure = cycle_dir / 'snapshot_failure.json'
            stale_failure.unlink(missing_ok=True)
            if final_dir.exists():
                archive_dir = cycle_dir / 'archive'
                archive_dir.mkdir(parents=True, exist_ok=True)
                archive_target = archive_dir / f'final_outputs-{run_id}'
                suffix = 1
                while archive_target.exists():
                    archive_target = archive_dir / f'final_outputs-{run_id}-{suffix}'
                    suffix += 1
                shutil.move(str(final_dir), str(archive_target))
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)

            # This is deliberately independent from all topology work.
            try:
                self._save_su_distribution_only(nodes, str(temp_dir))
            except Exception as exc:
                errors.append({
                    'artifact': 'final_su_distribution.csv',
                    'error': f'{type(exc).__name__}: {exc}',
                })

            try:
                self._save_final_node_exports(nodes=nodes, output_dir=str(temp_dir))
            except Exception as exc:
                errors.append({
                    'artifact': 'final_nodes_and_su_exports',
                    'error': f'{type(exc).__name__}: {exc}',
                })

            try:
                self._save_final_result_summary(
                    nodes=nodes,
                    E_target=E_target,
                    S_target=S_target,
                    output_dir=str(temp_dir),
                    raise_topology_errors=True,
                )
            except Exception as exc:
                errors.append({
                    'artifact': 'final_topology_specs.csv',
                    'error': f'{type(exc).__name__}: {exc}',
                })

            pending = temp_dir / 'final_pending_topology_specs.csv'
            if not pending.exists():
                pd.DataFrame(columns=[
                    'actual_su', 'gid', 'pending_kind', 'phase', 'contract_id',
                    'source_fragment_ids', 'claimed_source_node_ids',
                    'required_counts', 'available_counts', 'reason',
                ]).to_csv(pending, index=False)

            terminal = dict(
                (stage3_meta or {}).get('terminal_diagnostics', {}) or {}
            )
            completion = dict(
                (stage3_meta or {}).get('completion_diagnostics', {}) or {}
            )
            active_snapshot = completion if bool(stage3_complete) else terminal
            active_products = [
                dict(item or {})
                for item in list(active_snapshot.get('products', []) or [])
                if dict(item or {})
            ]
            active_pending = [
                dict(item or {})
                for item in list(active_snapshot.get('pending_contracts', []) or [])
                if dict(item or {})
            ]

            # A topology audit can fail after Stage3 has already committed
            # products.  Publish those products and the live pending contracts
            # independently, so an incomplete cycle remains inspectable.
            if not (temp_dir / 'final_topology_specs.csv').exists():
                topology_columns = [
                    'actual_su', 'gid', 'product_type',
                    'required_slots_satisfied', 'source_fragment_ids',
                    'fixed_source_node_ids', 'claimed_resource_ids',
                ]
                pd.DataFrame(
                    self._compact_topology_export_rows(active_products),
                    columns=topology_columns,
                ).to_csv(temp_dir / 'final_topology_specs.csv', index=False)
            if (not bool(stage3_complete)) or not (temp_dir / 'final_pending_topology_specs.csv').exists():
                pending_frame = pd.DataFrame(
                    self._compact_pending_export_rows(active_pending),
                    columns=[
                        'actual_su', 'gid', 'pending_kind', 'phase',
                        'contract_id', 'source_fragment_ids',
                        'claimed_source_node_ids', 'required_counts',
                        'available_counts', 'reason',
                    ],
                )
                pending_frame.to_csv(
                    temp_dir / 'final_pending_topology_specs.csv', index=False
                )

            proof = dict(
                terminal.get('infeasibility_proof', {})
                or completion.get('residual_infeasibility_proof', {})
                or {}
            )
            phase = str(
                terminal.get('phase', '')
                or completion.get('phase', '')
                or dict(completion.get('phase_result', {}) or {}).get('phase', '')
                or (stage3_meta or {}).get('phase', '')
                or ('complete' if bool(stage3_complete) else '')
                or 'unknown'
            )
            stop_reason = str(
                terminal.get('stop_reason', '')
                or (stage3_meta or {}).get('stop_reason', '')
                or ('stage3_complete' if stage3_complete else 'stage3_incomplete')
            )
            node_histogram = self._histogram_from_nodes(nodes).detach().cpu().long()
            expected_histogram = torch.clamp(
                H.detach().cpu().long(), min=0
            ).flatten()
            compare_size = max(
                int(node_histogram.numel()), int(expected_histogram.numel())
            )
            node_histogram_compare = torch.zeros(compare_size, dtype=torch.long)
            stage_histogram_compare = torch.zeros(compare_size, dtype=torch.long)
            node_histogram_compare[:int(node_histogram.numel())] = (
                node_histogram.flatten()
            )
            stage_histogram_compare[:int(expected_histogram.numel())] = (
                expected_histogram
            )
            histogram_matches = bool(torch.equal(
                node_histogram_compare, stage_histogram_compare
            ))
            if not histogram_matches:
                errors.append({
                    'artifact': 'su_identity',
                    'error': 'node_histogram_does_not_match_stage_H',
                })

            ownership_snapshot = dict(
                active_snapshot.get('ownership_ledger', {}) or {}
            )
            state_by_node: Dict[int, str] = {}
            for state, raw_ids in dict(
                ownership_snapshot.get('states', {}) or {}
            ).items():
                for raw_id in list(raw_ids or []):
                    state_by_node[int(raw_id)] = str(state)
            resource_states = {
                'remaining_closure_resource',
                'remaining_su24_ring_resource',
                'remaining_aromatic_donor',
            }
            ownership_rows = []
            for node in list(nodes or []):
                node_id = int(getattr(node, 'global_id', -1))
                if node_id < 0:
                    continue
                state = str(state_by_node.get(node_id, 'unclassified_or_conflict'))
                owner_product = ''
                for product in active_products:
                    owned_ids = self._owned_product_node_ids(product)
                    if node_id in set(owned_ids):
                        owner_product = str(product.get('product_id', '') or '')
                        break
                ownership_rows.append({
                    'node_id': node_id,
                    'su_type': int(getattr(node, 'su_type', -1)),
                    'state': state,
                    'resource_pool': bool(state in resource_states),
                    'product_id': owner_product,
                })
            if not (temp_dir / 'final_topology_node_ownership.csv').exists():
                pd.DataFrame(ownership_rows).to_csv(
                    temp_dir / 'final_topology_node_ownership.csv', index=False
                )
            if not (temp_dir / 'final_topology_resource_pool.csv').exists():
                pd.DataFrame(
                    [row for row in ownership_rows if bool(row['resource_pool'])],
                    columns=['node_id', 'su_type', 'state', 'resource_pool', 'product_id'],
                ).to_csv(temp_dir / 'final_topology_resource_pool.csv', index=False)
            resource_summary_rows = []
            for row in ownership_rows:
                resource_summary_rows.append({
                    'node_id': int(row['node_id']),
                    'su_type': int(row['su_type']),
                    'state': str(row['state']),
                    'resource_pool': bool(row['resource_pool']),
                    'product_id': str(row['product_id']),
                })
            pd.DataFrame(resource_summary_rows).to_csv(
                temp_dir / 'final_resource_usage_summary.csv', index=False
            )
            usage_counts: Dict[Tuple[int, str], int] = {}
            for row in ownership_rows:
                key = (int(row['su_type']), str(row['state']))
                usage_counts[key] = int(usage_counts.get(key, 0)) + 1
            pd.DataFrame([
                {
                    'su_type': int(su_type),
                    'state': str(state),
                    'count': int(count),
                    'resource_pool': bool(state in resource_states),
                }
                for (su_type, state), count in sorted(usage_counts.items())
            ], columns=['su_type', 'state', 'count', 'resource_pool']).to_csv(
                temp_dir / 'final_resource_usage_counts.csv', index=False
            )

            def _finite_or_none(value: Any) -> Optional[float]:
                try:
                    number = float(value)
                except Exception:
                    return None
                return number if np.isfinite(number) else None

            status = {
                **result,
                'published': True,
                'complete': bool(stage3_complete),
                'snapshot_ok': not bool(errors),
                'phase': phase,
                'stop_reason': stop_reason,
                'pre_layer3_r2': _finite_or_none(pre_layer3_r2),
                'post_layer3_r2': _finite_or_none(post_layer3_r2),
                'selected_r2': _finite_or_none(
                    dict(selected_diff or {}).get('r2', float('nan'))
                ),
                'spectral_contract': str(
                    dict(selected_diff or {}).get('spectral_contract', '') or ''
                ),
                'layer3_committed': bool(layer3_committed),
                'terminal_proof': proof,
                'stage3_snapshot': {
                    'products': list(active_snapshot.get('products', []) or []),
                    'pending_contracts': list(
                        active_snapshot.get('pending_contracts', []) or []
                    ),
                    'ownership_ledger': dict(
                        active_snapshot.get('ownership_ledger', {}) or {}
                    ),
                    'resource_ledger': dict(
                        active_snapshot.get('resource_ledger', {}) or {}
                    ),
                },
                'stage3_meta': {
                    'accepted_cycles': int((stage3_meta or {}).get('accepted_cycles', 0) or 0),
                    'total_cycle_count': int(
                        dict((stage3_meta or {}).get('diagnostics', {}) or {}).get(
                            'total_cycle_count', 0
                        ) or 0
                    ),
                },
                'su_identity': {
                    'node_histogram': node_histogram.tolist(),
                    'stage_histogram': expected_histogram.tolist(),
                    'histogram_matches': bool(histogram_matches),
                    'node_su24_count': int(
                        node_histogram[24].item()
                        if int(node_histogram.numel()) > 24 else 0
                    ),
                    'stage_h_su24_count': int(
                        expected_histogram[24].item()
                        if int(expected_histogram.numel()) > 24 else 0
                    ),
                },
                'expected_paths': [
                    'final_su_distribution.csv',
                    'final_topology_specs.csv',
                    'final_pending_topology_specs.csv',
                    'final_topology_node_ownership.csv',
                    'final_topology_resource_pool.csv',
                    'final_resource_usage_summary.csv',
                    'final_resource_usage_counts.csv',
                    'outer_cycle_status.json',
                ],
            }
            for required_name in status['expected_paths'][:-1]:
                if not (temp_dir / required_name).exists():
                    errors.append({
                        'artifact': str(required_name),
                        'error': 'required_snapshot_artifact_missing',
                    })
            status['snapshot_ok'] = not bool(errors)
            status_path = temp_dir / 'outer_cycle_status.json'
            status_path.write_text(
                json.dumps(status, ensure_ascii=False, indent=2, default=str),
                encoding='utf-8',
            )

            temp_dir.replace(final_dir)
            result.update({
                'published': True,
                'status_path': str(final_dir / 'outer_cycle_status.json'),
                'output_dir': str(final_dir),
            })
            print(
                f"[Outer Cycle {int(outer_idx) + 1} Snapshot] "
                f"published={bool(result['published'])} "
                f"complete={bool(stage3_complete)} dir={final_dir}",
                flush=True,
            )
            print(f"  SU CSV: {final_dir / 'final_su_distribution.csv'}", flush=True)
            print(f"  topology CSV: {final_dir / 'final_topology_specs.csv'}", flush=True)
            if errors:
                failure_path = cycle_dir / 'snapshot_failure.json'
                failure_path.write_text(
                    json.dumps(
                        {
                            **result,
                            'published': True,
                            'snapshot_ok': False,
                            'errors': errors,
                        },
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    encoding='utf-8',
                )
                print(
                    f"[Outer Cycle {int(outer_idx) + 1} Snapshot] "
                    f"export_errors={len(errors)}",
                    flush=True,
                )
            return result
        except Exception as exc:
            failure = {
                **result,
                'published': False,
                'error': f'{type(exc).__name__}: {exc}',
            }
            failure_path = cycle_dir / 'snapshot_failure.json'
            failure_path.write_text(
                json.dumps(failure, ensure_ascii=False, indent=2, default=str),
                encoding='utf-8',
            )
            print(
                f"[Outer Cycle {int(outer_idx) + 1} Snapshot] failed: "
                f"{type(exc).__name__}: {exc}; continuing",
                flush=True,
            )
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            return failure

    def _audit_final_su_identity(self, nodes: List[_NodeV3]) -> Dict[str, Any]:
        """Verify product SU identities against the final graph and H."""
        node_by_id = {
            int(getattr(node, 'global_id', -1)): node
            for node in list(nodes or [])
            if int(getattr(node, 'global_id', -1)) >= 0
        }
        snapshot = dict(
            getattr(self, '_final_stage3_completion_diagnostics', {}) or {}
        )
        if not bool(snapshot.get('stage3_complete', False)):
            return {
                'ok': True,
                'audit_skipped': True,
                'reason': 'no_authoritative_stage3_completion_snapshot',
            }
        products = [
            dict(item or {}) for item in list(snapshot.get('products', []) or [])
        ]
        expected: Dict[int, int] = {}
        conflicts: List[Dict[str, Any]] = []
        for product_index, product in enumerate(products):
            explicit = dict(product.get('expected_su_type_by_node_id', {}) or {})
            if not explicit:
                for su_key, ids in dict(
                    product.get('claimed_resource_ids_by_su', {}) or {}
                ).items():
                    if not str(su_key).lstrip('-').isdigit():
                        continue
                    for node_id in list(ids or []):
                        explicit[str(node_id)] = int(su_key)
                for node_id in list(product.get('fixed_source_node_ids', []) or []):
                    current = node_by_id.get(int(node_id))
                    if current is not None:
                        explicit[str(node_id)] = int(getattr(current, 'su_type', -1))
            for raw_node_id, raw_su in explicit.items():
                node_id = int(raw_node_id)
                su_type = int(raw_su)
                prior = expected.get(node_id)
                if prior is not None and int(prior) != su_type:
                    conflicts.append({
                        'reason': 'product_expected_su_conflict',
                        'node_id': node_id,
                        'expected': [int(prior), su_type],
                        'product_index': int(product_index),
                    })
                expected[node_id] = su_type
        mismatches: List[Dict[str, Any]] = []
        for node_id, expected_su in sorted(expected.items()):
            node = node_by_id.get(int(node_id))
            if node is None:
                mismatches.append({
                    'reason': 'product_expected_node_missing',
                    'node_id': int(node_id),
                    'expected_su': int(expected_su),
                })
                continue
            actual_su = int(getattr(node, 'su_type', -1))
            if actual_su != int(expected_su):
                mismatches.append({
                    'reason': 'product_node_su_mismatch',
                    'node_id': int(node_id),
                    'expected_su': int(expected_su),
                    'actual_su': int(actual_su),
                })
        resource_ledger = dict(snapshot.get('resource_ledger', {}) or {})
        recycled_su24_ids = {
            int(value)
            for value in list(resource_ledger.get('recycled_su24_to_23_ids', []) or [])
        }
        recovered_resource_ids = {
            int(value)
            for values in dict(
                resource_ledger.get('released_resource_ids_by_su', {}) or {}
            ).values()
            for value in list(values or [])
        }
        recovered_resource_ids.update(recycled_su24_ids)
        for node_id in sorted(recycled_su24_ids):
            node = node_by_id.get(int(node_id))
            if node is not None and int(getattr(node, 'su_type', -1)) != 23:
                mismatches.append({
                    'reason': 'recycled_resource_not_su23',
                    'node_id': int(node_id),
                    'actual_su': int(getattr(node, 'su_type', -1)),
                })
        for node_id in sorted(recovered_resource_ids - set(expected)):
            if node_id in node_by_id:
                mismatches.append({
                    'reason': 'recycled_resource_not_assembled',
                    'node_id': int(node_id),
                })
        expected_su24_ids = {
            int(node_id) for node_id, su_type in expected.items() if int(su_type) == 24
        }
        actual_su24_ids = {
            int(node_id)
            for node_id, node in node_by_id.items()
            if int(getattr(node, 'su_type', -1)) == 24
        }
        # Keep this audit independent from a fully initialized pipeline.  The
        # topology tests intentionally construct a lightweight pipeline via
        # ``object.__new__``; SU identity itself only depends on the nodes.
        h_su24_count = sum(
            1
            for node in list(nodes or [])
            if int(getattr(node, 'su_type', -1)) == 24
        )
        if expected_su24_ids != actual_su24_ids:
            mismatches.append({
                'reason': 'su24_identity_set_mismatch',
                'expected_su24_ids': sorted(expected_su24_ids),
                'actual_su24_ids': sorted(actual_su24_ids),
            })
        if int(h_su24_count) != len(actual_su24_ids):
            mismatches.append({
                'reason': 'su24_histogram_count_mismatch',
                'h_su24_count': int(h_su24_count),
                'node_su24_count': int(len(actual_su24_ids)),
            })
        return {
            'ok': not bool(conflicts or mismatches),
            'expected_su24_ids': sorted(expected_su24_ids),
            'actual_su24_ids': sorted(actual_su24_ids),
            'h_su24_count': int(h_su24_count),
            'product_expected_node_count': int(len(expected)),
            'recycled_su24_to_23_ids': sorted(recycled_su24_ids),
            'recovered_resource_ids': sorted(recovered_resource_ids),
            'conflicts': conflicts,
            'mismatches': mismatches,
        }

    def _save_final_result_summary(self,
                                   nodes: List[_NodeV3],
                                   E_target: torch.Tensor,
                                   S_target: Optional[torch.Tensor],
                                   output_dir: str,
                                   *,
                                   raise_topology_errors: bool = False) -> None:
        consistency = self._audit_final_su_identity(nodes)
        if not bool(consistency.get('ok', False)):
            raise RuntimeError(
                'final_su_identity_mismatch:'
                + json.dumps(consistency, ensure_ascii=False, sort_keys=True)
            )
        final_dir = Path(output_dir)
        final_dir.mkdir(parents=True, exist_ok=True)

        H_final = self._histogram_from_nodes(nodes).detach().cpu().long()
        E_pred_effective = get_effective_nodes_element_vector(
            nodes,
            self.E_SU.detach().cpu(),
            device=torch.device('cpu'),
        )
        E_pred_raw = get_effective_hist_element_vector(
            H_final,
            E_SU_tensor=self.E_SU.detach().cpu(),
            device=torch.device('cpu'),
        )
        e_target = E_target.detach().cpu().float().flatten()
        elem_names = ['C', 'H', 'O', 'N', 'S', 'X']
        rows = []
        for idx, name in enumerate(elem_names):
            tgt = float(e_target[idx].item()) if idx < int(e_target.numel()) else 0.0
            pred = float(E_pred_effective[idx].item()) if idx < int(E_pred_effective.numel()) else 0.0
            raw_pred = float(E_pred_raw[idx].item()) if idx < int(E_pred_raw.numel()) else 0.0
            diff = float(pred - tgt)
            rel = float(abs(diff) / tgt) if tgt > 1e-8 else 0.0
            rows.append({
                'element': str(name),
                'predicted': float(pred),
                'raw_hist_predicted': float(raw_pred),
                'target': float(tgt),
                'diff': float(diff),
                'rel_error': float(rel),
            })
        pd.DataFrame(rows).to_csv(final_dir / 'final_element_composition.csv', index=False)

        try:
            topology_rows, topology_summary = self._final_topology_rows(
                nodes,
                S_target=S_target,
                E_target=E_target,
            )
            debug_dir = final_dir / 'debug'
            debug_dir.mkdir(parents=True, exist_ok=True)
            topology_columns = [
                'actual_su', 'gid', 'product_type', 'required_slots_satisfied',
                'source_fragment_ids', 'fixed_source_node_ids', 'claimed_resource_ids',
            ]
            pd.DataFrame(
                self._compact_topology_export_rows(topology_rows),
                columns=topology_columns,
            ).to_csv(final_dir / 'final_topology_specs.csv', index=False)
            with open(debug_dir / 'final_topology_slot_details.jsonl', 'w', encoding='utf-8') as handle:
                for row in topology_rows:
                    handle.write(json.dumps(dict(row or {}), ensure_ascii=False, default=str) + '\n')
            ring_details: List[Dict[str, Any]] = []
            for index, raw_row in enumerate(topology_rows):
                row = dict(raw_row or {})
                audit = dict(row.get('ring_audit', {}) or {})
                if not audit:
                    continue
                ring_details.append({
                    'ring_index': int(index) + 1,
                    'product_id': str(row.get('product_id', '')),
                    'phase': str(row.get('phase', '')),
                    'product_type': str(row.get('product_type', '')),
                    'topology_notation': str(row.get('topology_notation', '')),
                    'node_id_notation': str(row.get('node_id_notation', '')),
                    **audit,
                })
            ring_columns = [
                'ring_index', 'generation_id', 'phase', 'product_type',
                'ring_kind', 'topology_notation', 'node_id_notation',
                'cycles', 'like24_node_ids', 'node_profiles_by_id',
                'external_attachments', 'branch_paths',
                'claimed_resource_ids_by_su', 'complete', 'audit_reason',
                'candidate_signature',
            ]
            pd.DataFrame([
                {
                    **detail,
                    'cycles': json.dumps(detail.get('cycles', []), ensure_ascii=False),
                    'like24_node_ids': json.dumps(
                        detail.get('like24_node_ids', []), ensure_ascii=False
                    ),
                    'node_profiles_by_id': json.dumps(
                        detail.get('node_profiles_by_id', {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    'external_attachments': json.dumps(
                        detail.get('external_attachments', []),
                        ensure_ascii=False,
                    ),
                    'branch_paths': json.dumps(
                        detail.get('branch_paths', []),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    'claimed_resource_ids_by_su': json.dumps(
                        detail.get('claimed_resource_ids_by_su', {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
                for detail in ring_details
            ], columns=ring_columns).to_csv(
                final_dir / 'final_stage3_ring_summary.csv', index=False
            )
            with open(
                debug_dir / 'final_stage3_ring_audit.jsonl',
                'w',
                encoding='utf-8',
            ) as handle:
                for detail in ring_details:
                    handle.write(
                        json.dumps(detail, ensure_ascii=False, default=str) + '\n'
                    )
            topo_diag = dict(topology_summary.get('diagnostics', {}) or {})
            topo_scalar = {
                'ok': bool(topology_summary.get('ok', False)),
                'topology_plan_feasible': bool(topology_summary.get('topology_plan_feasible', False)),
                'required_slots_satisfied_count': int(topology_summary.get('closed_topology_count', 0)),
                'final_topology_count': int(topology_summary.get('final_topology_count', 0)),
                'pending_closure_total_count': int(topology_summary.get('pending_closure_total_count', 0)),
                'solver_reason': str(topo_diag.get('solver_reason', '')),
                'stage3_terminal_stop_reason': str(
                    topo_diag.get('stage3_terminal_stop_reason', '') or ''
                ),
                'stage3_terminal_phase': str(
                    topo_diag.get('stage3_terminal_phase', '') or ''
                ),
                'stage3_terminal_snapshot_match': bool(
                    topo_diag.get('stage3_terminal_snapshot_match', False)
                ),
            }
            resource_ledger = dict(topo_diag.get('resource_ledger', {}) or {})
            initial_counts = {
                str(key): int(value)
                for key, value in dict(resource_ledger.get('initial_counts', {}) or {}).items()
            }
            topology_consumed_counts = {
                str(key): int(value)
                for key, value in dict(resource_ledger.get('topology_consumed_counts', {}) or {}).items()
            }
            residual_consumed_counts = {
                str(key): int(value)
                for key, value in dict(resource_ledger.get('residual_consumed_counts', {}) or {}).items()
            }
            remaining_counts = {
                str(key): int(value)
                for key, value in dict(resource_ledger.get('remaining_counts', {}) or {}).items()
            }
            closure_keys = ('11', '22', '23')
            ring_resource_key = '24'
            aromatic_donor_keys = ('10', '12', '13')
            topo_scalar.update({
                'resource_snapshot_stage': str(resource_ledger.get('snapshot_stage', '')),
                'initial_closure_resource_count': int(sum(
                    initial_counts.get(key, 0) for key in closure_keys
                )),
                'topology_consumed_closure_resource_count': int(sum(
                    topology_consumed_counts.get(key, 0) for key in closure_keys
                )),
                'residual_consumed_closure_resource_count': int(sum(
                    residual_consumed_counts.get(key, 0) for key in closure_keys
                )),
                'remaining_closure_resource_count': int(sum(
                    remaining_counts.get(key, 0) for key in closure_keys
                )),
                'initial_su24_ring_resource_count': int(
                    initial_counts.get(ring_resource_key, 0)
                ),
                'topology_consumed_su24_ring_resource_count': int(
                    topology_consumed_counts.get(ring_resource_key, 0)
                ),
                'residual_consumed_su24_ring_resource_count': int(
                    residual_consumed_counts.get(ring_resource_key, 0)
                ),
                'remaining_su24_ring_resource_count': int(
                    remaining_counts.get(ring_resource_key, 0)
                ),
                'remaining_aromatic_donor_count': int(sum(
                    remaining_counts.get(key, 0) for key in aromatic_donor_keys
                )),
                'remaining_resource_pool_count': int(sum(remaining_counts.values())),
            })
            unclosed_rows = list(topology_summary.get('unclosed_candidates', []) or [])
            pending_columns = [
                'actual_su', 'gid', 'pending_kind', 'phase', 'contract_id',
                'source_fragment_ids', 'claimed_source_node_ids', 'required_counts',
                'available_counts', 'reason',
            ]
            pending_frame = pd.DataFrame(
                self._compact_pending_export_rows(unclosed_rows),
                columns=pending_columns,
            )
            pending_frame.to_csv(
                final_dir / 'final_pending_topology_specs.csv', index=False
            )
            (final_dir / 'final_topology_unclosed_candidates.csv').unlink(missing_ok=True)
            with open(debug_dir / 'final_pending_topology_details.jsonl', 'w', encoding='utf-8') as handle:
                for row in unclosed_rows:
                    handle.write(json.dumps(dict(row or {}), ensure_ascii=False, default=str) + '\n')
            ownership_rows, ownership_conflicts = self._topology_ownership_rows(
                nodes=nodes,
                topology_rows=topology_rows,
                pending_rows=unclosed_rows,
                summary=topology_summary,
            )
            pd.DataFrame(ownership_rows).to_csv(
                final_dir / 'final_topology_node_ownership.csv',
                index=False,
            )
            resource_rows = [
                {
                    'node_id': int(row.get('node_id', -1)),
                    'su_type': int(row.get('su_type', -1)),
                    'state': str(row.get('state', '')),
                }
                for row in ownership_rows
                if bool(row.get('resource_pool', False))
            ]
            pd.DataFrame(resource_rows, columns=['node_id', 'su_type', 'state']).to_csv(
                final_dir / 'final_topology_resource_pool.csv',
                index=False,
            )
            topo_scalar.update({
                'total_node_count': int(len(ownership_rows)),
                'ownership_conflict_count': int(len(ownership_conflicts)),
                'unclassified_or_conflict_count': int(sum(
                    1 for row in ownership_rows if row.get('state') == 'unclassified_or_conflict'
                )),
                'remaining_pool_node_count': int(len(resource_rows)),
                'coverage_valid': bool(topology_summary.get('coverage_valid', False)),
            })
            pd.DataFrame([topo_scalar]).to_csv(final_dir / 'final_topology_summary.csv', index=False)
            with open(debug_dir / 'final_topology_diagnostics.json', 'w', encoding='utf-8') as handle:
                json.dump(
                    {
                        'summary': topology_summary,
                        'ownership_conflicts': ownership_conflicts,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            reject_rows = list(topology_summary.get('classification_rejects', []) or [])
            reject_columns = sorted({str(key) for row in reject_rows for key in dict(row or {}).keys()}) or [
                'node_id', 'su_type', 'family', 'reason'
            ]
            pd.DataFrame(reject_rows, columns=reject_columns).to_csv(
                final_dir / 'final_topology_classification_rejects.csv',
                index=False,
            )
        except Exception as e:
            print(f"[Final Topology Export] 保存失败: {e}")
            if raise_topology_errors:
                raise
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def reconstruct_spectrum(self, nodes: List[_NodeV3]) -> torch.Tensor:
        """Reconstruct the inverse spectrum with unit peaks and HWHM 1.0."""
        device = self.device
        ppm_axis = PPM_AXIS.to(device)

        mus = []
        pis = []
        peak_scales = []
        for n in nodes:
            try:
                center_su = int(n.su_type)
            except Exception:
                continue
            try:
                is_carbon = float(self.E_SU[center_su, 0].detach().cpu().item()) > 0
            except Exception:
                is_carbon = False
            if not bool(is_carbon):
                continue

            try:
                mu = float(getattr(n, 'mu', 0.0))
            except Exception:
                continue
            if float(mu) == 0.0:
                continue
            mus.append(float(mu))
            pis.append(1.0)
            peak_scales.append(
                reconstruction_peak_scale(
                    center_su,
                    self.aromatic_response_scale,
                )
            )

        if not mus:
            return torch.zeros_like(ppm_axis)

        mu_t = torch.tensor(mus, dtype=torch.float, device=device)
        pi_t = torch.tensor(pis, dtype=torch.float, device=device)
        pi_t = pi_t * torch.tensor(
            peak_scales,
            dtype=torch.float,
            device=device,
        )
        return lorentzian_spectrum(mu_t, pi_t, ppm_axis, hwhm=1.0)

# ============================================================================
# 工具函数（用于外部调用）
# ============================================================================

def _extract_checkpoint_state_dict(ckpt: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return dict(ckpt["model_state_dict"]), dict(ckpt)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return dict(ckpt["state_dict"]), dict(ckpt)
    return dict(ckpt), {"legacy_state_dict_only": True}


def _normalize_g2s_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in dict(state_dict or {}).items():
        key_s = str(key)
        if key_s.startswith("decoder.decoder."):
            key_s = "decoder." + key_s[len("decoder.decoder."):]
        normalized[key_s] = value
    return normalized


def _g2s_predict_intensity_from_state_dict(state_dict: Dict[str, torch.Tensor],
                                           fallback: bool = False) -> bool:
    weight = state_dict.get("decoder.net.5.weight")
    if weight is None:
        return bool(fallback)
    try:
        return bool(int(weight.shape[0]) > 1)
    except Exception:
        return bool(fallback)


def load_pipeline(s2n_ckpt: str, vae_ckpt: str, templates_pkl: str,
                  device: str = 'cuda') -> InversePipelineV3:
    """加载完整推理管道，兼容新旧 S2N/G2S checkpoint 格式。"""
    s2n_ckpt_obj = torch.load(s2n_ckpt, map_location=device, weights_only=False)
    s2n_state, s2n_meta = _extract_checkpoint_state_dict(s2n_ckpt_obj)
    s2n_model = S2NModel.from_config(S2NConfig.from_dict(s2n_meta.get("config")))
    s2n_model.load_state_dict(s2n_state)

    vae_ckpt_obj = torch.load(vae_ckpt, map_location=device, weights_only=False)
    vae_state_raw, vae_meta = _extract_checkpoint_state_dict(vae_ckpt_obj)
    vae_state = _normalize_g2s_state_dict(vae_state_raw)
    g2s_cfg = G2SConfig.from_dict(vae_meta.get("config"))
    g2s_cfg.predict_intensity = _g2s_predict_intensity_from_state_dict(
        vae_state,
        fallback=bool(getattr(g2s_cfg, "predict_intensity", False)),
    )
    vae_model = NMR_VAE.from_config(g2s_cfg)
    vae_model.load_state_dict(vae_state)

    template_path = str(Path(templates_pkl))
    pipeline = InversePipelineV3(s2n_model, vae_model, template_path, device)
    return pipeline


def read_spectrum_csv(path: str) -> torch.Tensor:
    """从CSV读取谱图数据"""
    df = pd.read_csv(path, sep=r'[;, \t]', engine='python', header=None)
    if df.shape[1] < 2:
        raise ValueError("CSV需要两列: ppm, intensity")
    ppm = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    intensity = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
    return resample_spectrum_to_ppm_axis(ppm, intensity, ppm_axis=PPM_AXIS)


def parse_elements(expr: str) -> torch.Tensor:
    """解析元素表达式，如 'C=100 H=150 O=10 N=2'"""
    import re
    matches = dict(re.findall(r"([CHONSX])\s*=\s*(\d+)", expr.upper()))
    return torch.tensor([int(matches.get(sym, 0)) for sym in ['C', 'H', 'O', 'N', 'S', 'X']], 
                       dtype=torch.float)


# ============================================================================
# 命令行接口（用于测试）
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='InversePipelineV3: 改进的逆向推理')
    parser.add_argument('--s2n_ckpt', type=str, required=True, help='S2N模型检查点路径')
    parser.add_argument('--vae_ckpt', type=str, required=True, help='VAE模型检查点路径')
    parser.add_argument('--templates', type=str, required=True, help='模板库pkl路径')
    parser.add_argument('--spectrum', type=str, required=True, help='目标谱图CSV路径')
    parser.add_argument('--elements', type=str, required=True, help='元素组成，如 C=100 H=150 O=10')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--output', type=str, default='inverse_result', help='输出目录')
    
    args = parser.parse_args()
    
    # 加载管道
    print(f"加载模型: S2N={args.s2n_ckpt}, VAE={args.vae_ckpt}")
    pipeline = load_pipeline(args.s2n_ckpt, args.vae_ckpt, args.templates, args.device)
    
    # 读取输入
    print(f"读取谱图: {args.spectrum}")
    S_target_raw = read_spectrum_csv(args.spectrum)
    
    print(f"解析元素: {args.elements}")
    E_target = parse_elements(args.elements)
    S_target = normalize_spectrum_to_carbon_count(S_target_raw, float(E_target[0].item()))
    
    print(f"目标谱图维度: {S_target.shape}, 元素组成: {E_target.tolist()}")
    
    # 执行推理
    nodes, H_final = pipeline.infer(
        S_target,
        E_target,
        save_intermediates=True,
        output_dir=args.output,
        eval_lib_path=args.templates,
    )
    
    print(f"\n推理完成！结果已保存至 {args.output}/")
    print(f"最终SU数量: {len(nodes)}")
    print(f"SU直方图: {H_final.long().tolist()}")
