"""Spectrum-to-node inventory model.

S2N estimates the global abundance of the 33 coarse-grained structural
units from a normalized 13C NMR spectrum and elemental composition. It is
used as an initialization model for the inverse reconstruction pipeline,
not as the final chemistry validator.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from ...shared.inverse_common import E_SU, NUM_SU_TYPES, SU_DEFS
except ImportError:
    from model.shared.inverse_common import E_SU, NUM_SU_TYPES, SU_DEFS


_SU_NAME_TO_FREQ = {
    "Carbocyclic_Aro_CH": 23652.,
    "Alkyl_CH2": 15152.,
    "Alkyl_CH3": 10518.,
    "Alcohol_Ether_C": 5118.,
    "Alkyl_Substituted_Aro_C": 3982.,
    "Halogen_X": 3233.,
    "Hydroxyl_O": 2579.,
    "Ether_O": 2486.,
    "O_Substituted_Aro_C": 2337.,
    "Amine_C": 2282.,
    "Alkyl_CH": 2057.,
    "Amine_Nitrogen": 1883.,
    "X_Substituted_Aro_C": 1839.,
    "Aromatic_Bridgehead_C": 1819.,
    "Ester_Group": 1797.,
    "Keto_Substituted_Aro_C": 1698.,
    "Vinyllic_CH": 1695.,
    "Aldehyde_Ketone_C": 1504.,
    "Heterocyclic_N": 1198.,
    "Carboxylic_Acid": 1134.,
    "N_Substituted_Aro_C": 1100.,
    "Amide_Group": 953.,
    "Halogenated_C": 802.,
    "Alkyl_Cq": 759.,
    "Vinyllic_Cq": 591.,
    "Aryl_Substituted_Aro_C": 560.,
    "Thioether_S": 527.,
    "Vinyllic_CH2": 414.,
    "Nitrile_C": 334.,
    "S_Substituted_Aro_C": 260.,
    "Alkynyl_Cq": 259.,
    "Heterocyclic_S": 167.,
    "Alkynyl_CH": 65.,
}

SU_FREQ = torch.tensor([_SU_NAME_TO_FREQ[name] for name, _ in SU_DEFS], dtype=torch.float)


SU_GROUPS: dict[str, list[int]] = {
    "carbonyl": [0, 1, 2, 3, 4],
    "aromatic": list(range(5, 14)),
    "unsaturated": list(range(14, 19)),
    "aliphatic": list(range(19, 26)),
    "hetero_terminal": list(range(26, 33)),
    "oxygen_related": [0, 1, 2, 3, 5, 19, 28, 29],
    "nitrogen_related": [0, 4, 6, 20, 26, 27],
    "sulfur_related": [7, 30, 31],
    "halogen_related": [8, 21, 32],
}

SU_REQUIRED_ELEMENT_INDICES: dict[str, list[int]] = {
    "O": SU_GROUPS["oxygen_related"],
    "N": SU_GROUPS["nitrogen_related"],
    "S": SU_GROUPS["sulfur_related"],
    "X": SU_GROUPS["halogen_related"],
}


@dataclass
class S2NConfig:
    hid: int = 256
    n_su_types: int = NUM_SU_TYPES
    base_channels: int = 64
    dropout: float = 0.10
    element_mode: str = "scaled_ratios"
    carbon_scale: float = 600.0
    use_spectrum_stats: bool = False
    spectrum_stats_mode: str = "regions_v1"
    target_mode: str = "per_100c"
    abundance_scale: float = 100.0
    count_loss: str = "smooth_l1"
    rare_threshold: float = 3000.0
    rare_max_weight: float = 5.0
    w_count: float = 1.0
    w_absolute: float = 0.30
    w_element: float = 0.35
    w_total: float = 0.25
    w_group: float = 0.20
    w_absent: float = 0.10
    w_high_count: float = 0.35
    high_count_threshold: float = 50.0
    high_count_weight: float = 4.0
    w_balance: float = 0.0
    balance_min_carbon: float = 50.0
    balance_target: str = "difference_per_100c"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "S2NConfig":
        if not data:
            return cls()
        data = dict(data)
        # Checkpoints created before the per-carbon target used absolute-count
        # logits. Preserve that behavior when loading old config dictionaries.
        if "target_mode" not in data:
            data["target_mode"] = "absolute"
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid})


def collect_pt_files(paths: Sequence[str | Path]) -> list[str]:
    files: list[str] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files.extend(str(p) for p in sorted(path.glob("*.pt")))
        elif path.is_file() and path.suffix == ".pt":
            files.append(str(path))
        else:
            raise FileNotFoundError(f"No .pt files found at '{path}'")
    if not files:
        raise FileNotFoundError(f"No .pt files found in: {paths}")
    return files


def normalize_element_vector(elements: torch.Tensor,
                             mode: str = "scaled_ratios",
                             carbon_scale: float = 600.0) -> torch.Tensor:
    """Convert raw [C,H,O,N,S,X] counts to the model input representation."""
    e = elements.float()
    if e.dim() == 1:
        e = e.unsqueeze(0)
    if mode == "raw":
        return e
    carbon = e[:, 0:1].clamp(min=1.0)
    ratios = e[:, 1:6] / carbon
    c_scaled = torch.log1p(carbon) / torch.log1p(torch.tensor(float(carbon_scale), device=e.device))
    return torch.cat([c_scaled, ratios], dim=-1)


def spectrum_stats_dim(mode: str = "regions_v1") -> int:
    if str(mode).lower() in {"none", "off", "disabled"}:
        return 0
    return 16


def extract_spectrum_stats(spectrum: torch.Tensor,
                           mode: str = "regions_v1") -> torch.Tensor:
    """Compute explicit NMR region statistics from the model input spectrum.

    The CNN branch can lose global area information because of normalization
    layers. These handcrafted statistics expose chemically meaningful region
    integrals directly to the fusion layer.
    """
    if str(mode).lower() in {"none", "off", "disabled"}:
        batch = int(spectrum.shape[0]) if spectrum.dim() > 1 else 1
        return spectrum.new_zeros((batch, 0))
    x = spectrum.float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    n = int(x.shape[-1])
    # Index mapping matches PPM_MIN=0.1, PPM_STEP=0.1 used by the dataset.
    def area(lo: float, hi: float) -> torch.Tensor:
        lo_i = max(0, min(n, int(float(lo) / 0.1)))
        hi_i = max(0, min(n, int(float(hi) / 0.1)))
        if hi_i <= lo_i:
            return x.new_zeros((x.shape[0],))
        return x[:, lo_i:hi_i].sum(dim=1) * 0.1

    total = x.sum(dim=1) * 0.1
    total_safe = total.clamp(min=1e-8)
    regions = [
        area(0.0, 25.0),
        area(25.0, 50.0),
        area(50.0, 70.0),
        area(70.0, 90.0),
        area(90.0, 110.0),
        area(110.0, 130.0),
        area(130.0, 160.0),
        area(160.0, 190.0),
        area(190.0, 240.0),
    ]
    frac = [r / total_safe for r in regions]
    aliphatic = area(0.0, 90.0) / total_safe
    aromatic = area(90.0, 160.0) / total_safe
    carbonyl = area(160.0, 240.0) / total_safe
    oxygenated = area(50.0, 90.0) / total_safe
    ordinary = area(0.0, 50.0) / total_safe
    log_area = torch.log1p(total).unsqueeze(1) / 8.0
    features = torch.stack(
        frac + [ordinary, oxygenated, aliphatic, aromatic, carbonyl],
        dim=1,
    )
    return torch.cat([features, log_area, (aromatic - aliphatic).unsqueeze(1)], dim=1)


def su_group_matrix(device: torch.device | None = None) -> torch.Tensor:
    names = ["carbonyl", "aromatic", "unsaturated", "aliphatic", "hetero_terminal"]
    mat = torch.zeros((len(names), NUM_SU_TYPES), dtype=torch.float)
    for row, name in enumerate(names):
        mat[row, SU_GROUPS[name]] = 1.0
    if device is not None:
        mat = mat.to(device)
    return mat


def rare_su_weights(config: S2NConfig, device: torch.device) -> torch.Tensor:
    freq = SU_FREQ.to(device)
    weights = torch.ones_like(freq)
    mask = freq < float(config.rare_threshold)
    if mask.any():
        min_freq = float(freq.min().item())
        denom = max(float(config.rare_threshold) - min_freq, 1.0)
        rare = 1.0 + (float(config.rare_threshold) - freq[mask]) / denom * (float(config.rare_max_weight) - 1.0)
        weights[mask] = torch.clamp(rare, min=1.0, max=float(config.rare_max_weight))
    return weights


def logits_to_su_counts(logits: torch.Tensor,
                        elements: torch.Tensor,
                        config: S2NConfig) -> torch.Tensor:
    """Convert model logits to absolute SU counts.

    In the default ``per_100c`` mode, the network learns SU abundance per
    100 carbon atoms. Absolute counts are then recovered by explicit carbon
    scaling. This keeps the model from compressing high-count merged samples
    toward the mean.
    """
    out = F.softplus(logits)
    if str(config.target_mode).lower() in {"per_100c", "per100c", "abundance"}:
        elem = elements.float().to(out.device)
        if elem.dim() == 1:
            elem = elem.unsqueeze(0)
        carbon = elem[:, 0:1].clamp(min=1.0)
        return out * carbon / float(config.abundance_scale)
    return out


def counts_to_training_target(counts: torch.Tensor,
                              elements: torch.Tensor,
                              config: S2NConfig) -> torch.Tensor:
    """Convert absolute SU counts to the model's primary training target."""
    y = counts.float()
    if str(config.target_mode).lower() in {"per_100c", "per100c", "abundance"}:
        elem = elements.float().to(y.device)
        if elem.dim() == 1:
            elem = elem.unsqueeze(0)
        carbon = elem[:, 0:1].clamp(min=1.0)
        return y / carbon * float(config.abundance_scale)
    return y


def apply_element_feasibility_mask(su_counts: torch.Tensor,
                                   elements: torch.Tensor,
                                   threshold: float = 0.5) -> torch.Tensor:
    """Zero SU categories that require absent hetero elements.

    This is a light feasibility mask for S2N predictions. It prevents
    impossible categories such as halogen nodes when X=0, while leaving
    detailed chemistry repair to downstream inverse reasoning.
    """
    pred = su_counts.clone()
    elem = elements.float().to(pred.device)
    if elem.dim() == 1:
        elem = elem.unsqueeze(0)
    squeeze = su_counts.dim() == 1
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    element_col = {"O": 2, "N": 3, "S": 4, "X": 5}
    for symbol, su_ids in SU_REQUIRED_ELEMENT_INDICES.items():
        absent = elem[:, element_col[symbol]] <= float(threshold)
        if absent.any():
            pred[absent][:, su_ids] = 0.0
    return pred.squeeze(0) if squeeze else pred


class ResidualSEBlock(nn.Module):
    """1D residual convolution block with squeeze-excitation attention."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1,
                 groups: int = 8, se_ratio: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=9, stride=stride, padding=4, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_c)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=9, padding=4, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_c)
        self.se_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc1 = nn.Linear(out_c, max(1, out_c // se_ratio))
        self.se_fc2 = nn.Linear(max(1, out_c // se_ratio), out_c)
        self.shortcut = (
            nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False)
            if stride != 1 or in_c != out_c
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        w = self.se_pool(h).squeeze(-1)
        w = torch.sigmoid(self.se_fc2(self.act(self.se_fc1(w))))
        return self.act(h * w.unsqueeze(-1) + res)


class SpectrumEncoderCNN(nn.Module):
    """Compact 1D ResNet-SE encoder for full 13C NMR spectra."""

    def __init__(self, in_channels: int = 1, hid: int = 256, base_c: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_c, kernel_size=15, stride=2, padding=7, bias=False),
            nn.GroupNorm(8, base_c),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualSEBlock(base_c, base_c * 2, stride=2)
        self.layer2 = ResidualSEBlock(base_c * 2, hid, stride=2)
        self.layer3 = ResidualSEBlock(hid, hid, stride=2)
        self.layer4 = ResidualSEBlock(hid, hid, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return self.pool(h).squeeze(-1)


class S2NModel(nn.Module):
    """Spectrum + elements -> non-negative SU inventory."""

    def __init__(self,
                 hid: int = 256,
                 n_su_types: int = NUM_SU_TYPES,
                 base_channels: int = 64,
                 dropout: float = 0.10,
                 element_mode: str = "scaled_ratios",
                 carbon_scale: float = 600.0,
                 use_spectrum_stats: bool = False,
                 spectrum_stats_mode: str = "regions_v1",
                 target_mode: str = "per_100c",
                 abundance_scale: float = 100.0):
        super().__init__()
        self.n_su_types = int(n_su_types)
        self.element_mode = str(element_mode)
        self.carbon_scale = float(carbon_scale)
        self.use_spectrum_stats = bool(use_spectrum_stats)
        self.spectrum_stats_mode = str(spectrum_stats_mode)
        self.spectrum_stats_dim = spectrum_stats_dim(self.spectrum_stats_mode) if self.use_spectrum_stats else 0
        self.target_mode = str(target_mode)
        self.abundance_scale = float(abundance_scale)
        self.spec_enc = SpectrumEncoderCNN(in_channels=1, hid=hid, base_c=base_channels)
        self.elem_fc = nn.Linear(6, hid)
        self.stats_fc = (
            nn.Sequential(
                nn.Linear(self.spectrum_stats_dim, hid // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hid // 2, hid // 2),
                nn.ReLU(inplace=True),
            )
            if self.spectrum_stats_dim > 0
            else None
        )
        fusion_in = hid * 2 + (hid // 2 if self.spectrum_stats_dim > 0 else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hid),
            nn.ReLU(inplace=True),
        )
        self.head_count = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hid, self.n_su_types),
        )

    @classmethod
    def from_config(cls, config: S2NConfig | dict | None) -> "S2NModel":
        cfg = config if isinstance(config, S2NConfig) else S2NConfig.from_dict(config)
        return cls(
            hid=cfg.hid,
            n_su_types=cfg.n_su_types,
            base_channels=cfg.base_channels,
            dropout=cfg.dropout,
            element_mode=cfg.element_mode,
            carbon_scale=cfg.carbon_scale,
            use_spectrum_stats=cfg.use_spectrum_stats,
            spectrum_stats_mode=cfg.spectrum_stats_mode,
            target_mode=cfg.target_mode,
            abundance_scale=cfg.abundance_scale,
        )

    def forward(self, spectrum: torch.Tensor, elements: torch.Tensor) -> torch.Tensor:
        if spectrum.dim() == 1:
            spectrum = spectrum.unsqueeze(0)
        elem_in = normalize_element_vector(
            elements,
            mode=self.element_mode,
            carbon_scale=self.carbon_scale,
        ).to(spectrum.device)
        h_spec = self.spec_enc(spectrum.unsqueeze(1))
        h_elem = F.relu(self.elem_fc(elem_in))
        features = [h_spec, h_elem]
        if self.stats_fc is not None:
            stats = extract_spectrum_stats(spectrum, mode=self.spectrum_stats_mode).to(spectrum.device)
            features.append(self.stats_fc(stats))
        h = self.fusion(torch.cat(features, dim=-1))
        return self.head_count(h)

    def infer_su_hist(self,
                      spectrum: torch.Tensor,
                      elements: torch.Tensor,
                      apply_mask: bool = False) -> torch.Tensor:
        with torch.no_grad():
            cfg = S2NConfig(target_mode=self.target_mode, abundance_scale=self.abundance_scale)
            pred = logits_to_su_counts(self.forward(spectrum, elements), elements, cfg)
            if apply_mask:
                pred = apply_element_feasibility_mask(pred, elements)
            return pred


class S2NDataset(Dataset):
    """Load S2N samples from processed graph .pt files."""

    def __init__(self, pt_files: Sequence[str | Path]):
        self.files = [str(p) for p in pt_files]
        if not self.files:
            raise ValueError("S2NDataset requires at least one .pt file.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        file_path = self.files[idx]
        data_dict = torch.load(file_path, map_location="cpu", weights_only=False)
        spectrum = data_dict.get("y_spectrum", data_dict.get("y_mol"))
        if spectrum is None:
            raise KeyError(f"{file_path} does not contain y_spectrum or y_mol.")
        return {
            "spectrum": spectrum.float(),
            "elements": data_dict["total_atom_counts"].float(),
            "n_su_true": data_dict["su_hist"].float(),
            "path": file_path,
        }


def s2n_loss(n_su_logits: torch.Tensor,
             n_su_true: torch.Tensor,
             elements_true: torch.Tensor,
             config: S2NConfig | None = None,
             **legacy_weights):
    """Composite S2N loss.

    Backward compatibility: legacy keyword weights w_cnt, w_elem_abs and
    w_nodesum are accepted and mapped to the new config weights.
    """
    cfg = config or S2NConfig()
    if legacy_weights:
        cfg = S2NConfig.from_dict(cfg.to_dict())
        if "w_cnt" in legacy_weights:
            cfg.w_count = float(legacy_weights["w_cnt"])
        if "w_elem_abs" in legacy_weights:
            cfg.w_element = float(legacy_weights["w_elem_abs"])
        if "w_nodesum" in legacy_weights:
            cfg.w_total = float(legacy_weights["w_nodesum"])

    pred = logits_to_su_counts(n_su_logits, elements_true, cfg)
    true = n_su_true.float()
    elements = elements_true.float()
    pred_target = F.softplus(n_su_logits)
    true_target = counts_to_training_target(true, elements, cfg)

    diff_target = torch.abs(pred_target - true_target)
    weights = rare_su_weights(cfg, pred.device).unsqueeze(0)
    high_mask = (true > float(cfg.high_count_threshold)).float()
    high_weights = 1.0 + high_mask * (float(cfg.high_count_weight) - 1.0)
    combined_weights = weights * high_weights
    if cfg.count_loss == "l1":
        loss_count = (diff_target * combined_weights).mean()
    else:
        loss_count = (F.smooth_l1_loss(pred_target, true_target, reduction="none") * combined_weights).mean()

    if cfg.count_loss == "l1":
        loss_absolute = torch.abs(pred - true).mean()
    else:
        loss_absolute = F.smooth_l1_loss(pred, true)

    e_su = E_SU.to(pred.device).float()
    elem_pred = pred @ e_su
    carbon = elements[:, 0:1].clamp(min=1.0)
    loss_element = F.smooth_l1_loss(elem_pred / carbon, elements / carbon)

    total_pred = pred.sum(dim=-1)
    total_true = true.sum(dim=-1)
    loss_total = F.smooth_l1_loss(total_pred, total_true)

    group_mat = su_group_matrix(pred.device)
    group_pred = pred @ group_mat.t()
    group_true = true @ group_mat.t()
    loss_group = F.smooth_l1_loss(group_pred / carbon, group_true / carbon)

    # Structural balance loss:
    # preserve the aromatic-vs-aliphatic resource contrast instead of only
    # matching their independent totals. This directly targets the observed
    # failure mode where aromatic carbon is shifted into aliphatic carbon.
    aromatic_ids = SU_GROUPS["aromatic"]
    aliphatic_ids = SU_GROUPS["aliphatic"]
    aro_pred = pred[:, aromatic_ids].sum(dim=1)
    aro_true = true[:, aromatic_ids].sum(dim=1)
    ali_pred = pred[:, aliphatic_ids].sum(dim=1)
    ali_true = true[:, aliphatic_ids].sum(dim=1)
    valid_balance = carbon.squeeze(1) >= float(cfg.balance_min_carbon)
    if valid_balance.any():
        if str(cfg.balance_target).lower() == "ratio":
            pred_balance = aro_pred[valid_balance] / ali_pred[valid_balance].clamp(min=1.0)
            true_balance = aro_true[valid_balance] / ali_true[valid_balance].clamp(min=1.0)
        else:
            pred_balance = (aro_pred[valid_balance] - ali_pred[valid_balance]) / carbon.squeeze(1)[valid_balance] * 100.0
            true_balance = (aro_true[valid_balance] - ali_true[valid_balance]) / carbon.squeeze(1)[valid_balance] * 100.0
        loss_balance = F.smooth_l1_loss(pred_balance, true_balance)
    else:
        loss_balance = pred.new_tensor(0.0)

    if high_mask.any():
        high_abs = torch.abs(pred - true) * high_mask
        loss_high_count = high_abs.sum() / high_mask.sum().clamp(min=1.0)
    else:
        loss_high_count = pred.new_tensor(0.0)

    loss_absent = pred.new_tensor(0.0)
    element_col = {"O": 2, "N": 3, "S": 4, "X": 5}
    for symbol, su_ids in SU_REQUIRED_ELEMENT_INDICES.items():
        absent = elements[:, element_col[symbol]] <= 0.5
        if absent.any():
            loss_absent = loss_absent + pred[absent][:, su_ids].mean()

    total = (
        cfg.w_count * loss_count
        + cfg.w_absolute * loss_absolute
        + cfg.w_element * loss_element
        + cfg.w_total * loss_total
        + cfg.w_group * loss_group
        + cfg.w_absent * loss_absent
        + cfg.w_high_count * loss_high_count
        + cfg.w_balance * loss_balance
    )
    parts = {
        "loss": total,
        "count": loss_count,
        "absolute": loss_absolute,
        "element": loss_element,
        "total_su": loss_total,
        "group": loss_group,
        "absent": loss_absent,
        "high_count": loss_high_count,
        "balance": loss_balance,
    }
    return total, parts
