"""Graph-to-shift model for centered coarse-grained k-hop subgraphs.

G2S predicts the 13C chemical shift of the center structural unit from its
local coarse-grained graph environment. The training target is the center-node
chemical shift only; intensity is intentionally not learned because it is less
stable and is better handled by node counts / deterministic spectrum
normalization in downstream inverse reconstruction.

For compatibility with the existing inverse code, the public class name
``NMR_VAE`` and the decoder interface are retained. The decoder returns two
columns ``[mu, pi_logit]`` when called directly, with ``pi_logit=0`` as a
neutral placeholder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GATv2Conv, GINEConv
from torch_geometric.utils import k_hop_subgraph, to_undirected

try:
    from ...shared.coarse_graph import NUM_SU_TYPES, SU_DEFS
except ImportError:
    from model.shared.coarse_graph import NUM_SU_TYPES, SU_DEFS


PPM_CENTER = 120.0
PPM_SCALE = 120.0

PPM_REGIONS: dict[str, tuple[float, float]] = {
    "alkyl": (0.0, 50.0),
    "hetero_aliphatic": (40.0, 100.0),
    "unsaturated": (70.0, 120.0),
    "aromatic": (90.0, 160.0),
    "carbonyl": (160.0, 240.0),
}

SU_GROUPS: dict[str, list[int]] = {
    "carbonyl": [0, 1, 2, 3, 4],
    "aromatic": list(range(5, 14)),
    "unsaturated": list(range(14, 19)),
    "aliphatic": list(range(19, 26)),
    "hetero_terminal": list(range(26, 33)),
}


@dataclass
class G2SConfig:
    k_hop: int = 3
    hid: int = 384
    latent_dim: int = 16
    n_node_feat: int = 39
    n_edge_feat: int = 2
    n_su_feat: int = NUM_SU_TYPES
    n_layers: int = 4
    g_dim: int = 2
    dropout: float = 0.10
    beta: float = 1.0e-4
    loss: str = "smooth_l1"
    predict_intensity: bool = False
    use_latent: bool = True
    use_global_element: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "G2SConfig":
        if not data:
            return cls()
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in dict(data).items() if k in valid})


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


def load_raw_data(path: str | Path | Sequence[str | Path]) -> list:
    """Load PyG molecular graph objects from a file, directory, or paths."""
    if isinstance(path, (list, tuple)):
        raw_list = []
        for p in path:
            raw_list.extend(load_raw_data(p))
        return raw_list

    path = Path(path)
    if path.is_dir():
        pt_files = sorted(path.glob("*.pt"))
        raw_list = []
        for file in pt_files:
            try:
                data = torch.load(file, weights_only=False)
                if isinstance(data, list):
                    for item in data:
                        _set_mol_attr(item, "source_path", str(file))
                    raw_list.extend(data)
                else:
                    _set_mol_attr(data, "source_path", str(file))
                    raw_list.append(data)
            except Exception as exc:
                print(f"  -> warning: failed to load '{file}': {exc}")
        print(f"Loaded {len(raw_list)} molecular .pt files from '{path}'")
        return raw_list

    if path.is_file():
        data = torch.load(path, weights_only=False)
        if isinstance(data, list):
            for item in data:
                _set_mol_attr(item, "source_path", str(path))
            return data
        _set_mol_attr(data, "source_path", str(path))
        return [data]

    raise FileNotFoundError(f"Path does not exist: {path}")


def _mol_get(mol, key: str, default=None):
    if isinstance(mol, dict):
        return mol.get(key, default)
    return getattr(mol, key, default)


def _set_mol_attr(mol, key: str, value):
    if isinstance(mol, dict):
        mol[key] = value
    else:
        try:
            setattr(mol, key, value)
        except Exception:
            pass


def _mol_id_from_path(path: str | Path) -> str:
    return Path(path).stem


class MyData(Data):
    """PyG data object with explicit batching behavior for center indices."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("center_id",):
            return self.x.size(0)
        if key in ("mol_id", "original_center_id"):
            return 0
        return super().__inc__(key, value, *args, **kwargs)


class LocalNMRDataset(InMemoryDataset):
    """Extract center-carbon k-hop subgraphs from molecular graphs."""

    def __init__(self, raw_list: Sequence, k: int = 3):
        self.k = int(k)
        super().__init__(".", transform=None, pre_transform=None)
        self.data, self.slices = self._process(raw_list)

    def _process(self, raw_list: Sequence):
        all_data: list[MyData] = []
        for mol_idx, mol in enumerate(raw_list):
            edge_index = _mol_get(mol, "edge_index")
            edge_attr = _mol_get(mol, "edge_attr", torch.zeros((0, 2), dtype=torch.float))
            x_node = _mol_get(mol, "x")
            is_carbon = _mol_get(mol, "is_carbon")
            y_mol = _mol_get(mol, "y_spectrum", _mol_get(mol, "y_mol"))
            real_ppm = _mol_get(mol, "real_ppm")
            real_intensity = _mol_get(mol, "real_intensity")
            su_hist_global = _mol_get(mol, "su_hist", torch.zeros(NUM_SU_TYPES))
            elem_counts_global = _mol_get(mol, "total_atom_counts", torch.zeros(6))
            smiles = _mol_get(mol, "smiles", "")
            source_path = _mol_get(mol, "source_path", "")

            if edge_index is None or x_node is None or is_carbon is None or real_ppm is None:
                continue

            if edge_attr is None:
                edge_attr = torch.zeros((edge_index.size(1), 2), dtype=torch.float, device=edge_index.device)
            if edge_index.numel() > 0:
                edge_index, edge_attr = to_undirected(edge_index, edge_attr)

            x_node = x_node.float()
            is_carbon = is_carbon.bool()
            real_ppm = real_ppm.float()
            real_intensity = real_intensity.float() if real_intensity is not None else torch.ones_like(real_ppm)

            total_nodes = su_hist_global.float().sum().clamp(min=1.0)
            total_atoms = elem_counts_global.float().sum().clamp(min=1.0)
            su_hist_freq = su_hist_global.float() / total_nodes
            elem_ratio = elem_counts_global.float() / total_atoms
            global_feat = torch.cat([su_hist_freq, elem_ratio], dim=0).view(1, -1)

            for center in range(x_node.size(0)):
                if not bool(is_carbon[center]):
                    continue
                ppm_value = float(real_ppm[center].item())
                if not (0.0 < ppm_value <= 260.0):
                    continue

                sub_nodes, sub_edge_index, mapping, mask = k_hop_subgraph(
                    center,
                    self.k,
                    edge_index,
                    relabel_nodes=True,
                    num_nodes=x_node.size(0),
                )

                data = MyData()
                data.x = x_node[sub_nodes]
                data.edge_index = sub_edge_index
                data.edge_attr = edge_attr[mask] if edge_attr.numel() > 0 else edge_attr
                data.original_center_id = torch.tensor([center], dtype=torch.long)
                data.center_id = torch.tensor([int(mapping)], dtype=torch.long)
                data.is_carbon = torch.tensor([True], dtype=torch.bool)
                data.mol_id = torch.tensor([mol_idx], dtype=torch.long)
                data.real_ppm = torch.tensor([ppm_value], dtype=torch.float)
                data.real_intensity = torch.tensor([float(real_intensity[center].item())], dtype=torch.float)
                data.center_su = x_node[center, :NUM_SU_TYPES].argmax().view(1).long()
                data.global_feat = global_feat
                if y_mol is not None:
                    data.y_mol = y_mol
                if smiles:
                    data.smiles = str(smiles)
                if source_path:
                    data.source_path = str(source_path)
                all_data.append(data)

        if not all_data:
            raise ValueError("No valid carbon-centered subgraphs were extracted.")
        return self.collate(all_data)


class G2SFileDataset(Dataset):
    """Lazy dataset wrapper that builds local subgraphs from individual files."""

    def __init__(self, files: Sequence[str | Path], k: int = 3):
        self.files = [str(p) for p in files]
        self.k = int(k)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        mol = torch.load(self.files[idx], weights_only=False)
        dataset = LocalNMRDataset([mol], k=self.k)
        data_list = dataset.to_data_list()
        for data in data_list:
            data.source_path = self.files[idx]
            data.source_mol = _mol_id_from_path(self.files[idx])
        return data_list


class EnvironmentEncoder(nn.Module):
    """GINE layers followed by GATv2 attention, returning center embeddings."""

    def __init__(self,
                 n_node_feat: int = 39,
                 n_edge_feat: int = 2,
                 hid: int = 384,
                 latent_dim: int = 16,
                 n_layers: int = 4,
                 dropout: float = 0.10):
        super().__init__()
        self.hid = int(hid)
        self.edge_enc = nn.Sequential(
            nn.Linear(n_edge_feat, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for idx in range(int(n_layers)):
            if idx < int(n_layers) - 1:
                mlp_in = n_node_feat if idx == 0 else hid
                mlp = nn.Sequential(nn.Linear(mlp_in, hid), nn.ReLU(), nn.Linear(hid, hid))
                conv = GINEConv(mlp, edge_dim=hid)
            else:
                heads = 8
                conv = GATv2Conv(hid, hid // heads, heads=heads, edge_dim=hid, dropout=dropout)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hid))

        self.head_mu = nn.Linear(hid, latent_dim)
        self.head_logvar = nn.Linear(hid, latent_dim)

    def forward(self, data):
        x, edge_index, edge_attr, center_id = data.x.float(), data.edge_index, data.edge_attr.float(), data.center_id
        if edge_attr.numel() == 0:
            e_emb = torch.zeros(edge_index.size(1), self.hid, dtype=x.dtype, device=x.device)
        else:
            e_emb = self.edge_enc(edge_attr)

        h = x
        for idx, conv in enumerate(self.convs):
            if idx == 0:
                h = conv(h, edge_index, edge_attr=e_emb)
            else:
                h_new = conv(h, edge_index, edge_attr=e_emb)
                h = h + h_new
            h = self.norms[idx](h)
            h = F.relu(h)

        center_vec = h[center_id]
        mu = self.head_mu(center_vec)
        logvar = self.head_logvar(center_vec)
        return mu, logvar, center_vec


class ShiftDecoder(nn.Module):
    """Decode center SU, latent environment and global composition into ppm."""

    def __init__(self,
                 n_su_feat: int = NUM_SU_TYPES,
                 latent_dim: int = 16,
                 g_dim: int = 2,
                 hid: int = 384,
                 dropout: float = 0.10,
                 out_dim: int = 1):
        super().__init__()
        self.out_dim = int(out_dim)
        in_dim = int(n_su_feat) + int(latent_dim) + int(g_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid // 2),
            nn.ReLU(),
            nn.Linear(hid // 2, self.out_dim),
        )

    def forward(self, su_feat: torch.Tensor, z: torch.Tensor, g_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([su_feat, z, g_embed], dim=-1)
        out = self.net(x)
        if out.shape[-1] == 1:
            zeros = torch.zeros_like(out)
            return torch.cat([out, zeros], dim=-1)
        return out


class NMRDecoder(ShiftDecoder):
    """Backward-compatible decoder name used by inverse reconstruction code."""


class NMR_VAE(nn.Module):
    """Centered k-hop graph -> center-node 13C chemical shift.

    The class name is retained for compatibility. In forward mode it returns:

        (mu_pred, pi_placeholder), (mu_latent, logvar_latent)

    ``pi_placeholder`` is a vector of ones. It is not trained and should not be
    treated as a learned intensity prediction.
    """

    def __init__(self,
                 n_node_feat: int = 39,
                 n_edge_feat: int = 2,
                 n_su_feat: int = NUM_SU_TYPES,
                 hid: int = 384,
                 latent_dim: int = 16,
                 g_dim: int = 2,
                 n_layers: int = 4,
                 dropout: float = 0.10,
                 predict_intensity: bool = False,
                 use_global_element: bool = True):
        super().__init__()
        self.encoder = EnvironmentEncoder(
            n_node_feat=n_node_feat,
            n_edge_feat=n_edge_feat,
            hid=hid,
            latent_dim=latent_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.global_mlp = nn.Sequential(nn.Linear(6, g_dim), nn.ReLU())
        self.decoder = NMRDecoder(
            n_su_feat=n_su_feat,
            latent_dim=latent_dim,
            g_dim=g_dim,
            hid=hid,
            dropout=dropout,
            out_dim=1 if not predict_intensity else 2,
        )
        self.latent_dim = int(latent_dim)
        self.predict_intensity = bool(predict_intensity)
        self.use_global_element = bool(use_global_element)

    @classmethod
    def from_config(cls, config: G2SConfig) -> "NMR_VAE":
        return cls(
            n_node_feat=config.n_node_feat,
            n_edge_feat=config.n_edge_feat,
            n_su_feat=config.n_su_feat,
            hid=config.hid,
            latent_dim=config.latent_dim,
            g_dim=config.g_dim,
            n_layers=config.n_layers,
            dropout=config.dropout,
            predict_intensity=config.predict_intensity,
            use_global_element=config.use_global_element,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def make_global_embedding(self, data) -> torch.Tensor:
        device = data.x.device
        batch_size = int(data.center_id.numel())
        if not self.use_global_element:
            return torch.zeros((batch_size, self.global_mlp[0].out_features), device=device)

        if hasattr(data, "global_feat"):
            g_raw_full = data.global_feat.to(device).float()
            if g_raw_full.dim() != 2:
                g_raw_full = g_raw_full.view(batch_size, -1)
            elem_part = g_raw_full[:, -6:]
            if elem_part.max() > 1.1:
                elem_part = elem_part / elem_part.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            elem_part = torch.zeros((batch_size, 6), dtype=torch.float, device=device)
        return 0.02 * self.global_mlp(elem_part)

    def forward(self, data):
        mu_latent, logvar_latent, _ = self.encoder(data)
        z = self.reparameterize(mu_latent, logvar_latent)
        center_node_x = data.x[data.center_id]
        su_feat = center_node_x[:, :NUM_SU_TYPES]
        g_embed = self.make_global_embedding(data)
        decoded = self.decoder(su_feat, z, g_embed)
        mu_pred = decoded[:, 0]
        if self.predict_intensity:
            pi_placeholder = F.softplus(decoded[:, 1])
        else:
            pi_placeholder = torch.ones_like(mu_pred)
        return (mu_pred, pi_placeholder), (mu_latent, logvar_latent)


def g2s_loss_function(pred_params,
                      latent_params,
                      true_data,
                      beta: float = 1.0e-4,
                      loss_type: str = "smooth_l1"):
    """Shift-only G2S loss on normalized ppm, with optional KL regularization."""
    mu_pred, _ = pred_params
    mu_latent, logvar_latent = latent_params
    mu_pred = mu_pred.view(-1)
    real_ppm = true_data.real_ppm.view(-1).to(mu_pred.device).float()
    is_carbon = true_data.is_carbon.view(-1).bool().to(mu_pred.device)
    mask = is_carbon & torch.isfinite(real_ppm) & (real_ppm > 0.0)
    if not mask.any():
        zero = (mu_pred * 0.0).sum()
        return zero, {"loss": zero, "mu": zero, "kl": zero, "mae_ppm": zero}

    pred_norm = (mu_pred - PPM_CENTER) / PPM_SCALE
    true_norm = (real_ppm - PPM_CENTER) / PPM_SCALE
    if loss_type == "l1":
        loss_mu = F.l1_loss(pred_norm[mask], true_norm[mask])
    else:
        loss_mu = F.smooth_l1_loss(pred_norm[mask], true_norm[mask])

    if beta > 0:
        kl = -0.5 * torch.sum(
            1 + logvar_latent[mask] - mu_latent[mask].pow(2) - logvar_latent[mask].exp(),
            dim=1,
        ).mean()
    else:
        kl = (mu_pred * 0.0).sum()

    total = loss_mu + float(beta) * kl
    mae_ppm = F.l1_loss(mu_pred[mask], real_ppm[mask])
    return total, {"loss": total, "mu": loss_mu, "kl": kl, "mae_ppm": mae_ppm}


def vae_loss_function(pred_params,
                      latent_params,
                      true_data,
                      beta: float = 1.0e-4,
                      lambda_mu: float = 1.0,
                      lambda_pi: float = 0.0):
    """Backward-compatible wrapper returning the historical 5-tuple.

    ``lambda_pi`` is accepted but ignored in shift-only training.
    """
    loss, parts = g2s_loss_function(pred_params, latent_params, true_data, beta=beta)
    zero = (parts["loss"] * 0.0)
    return loss * float(lambda_mu), parts["mu"] * float(lambda_mu), parts["kl"], parts["mae_ppm"], zero


@torch.no_grad()
def collect_val_regression_data(model, loader, device):
    """Compatibility helper returning mu arrays and dummy pi arrays."""
    model.eval()
    all_mu_true, all_mu_pred = [], []
    for data in loader:
        data = data.to(device)
        (mu_pred, _), _ = model(data)
        real_ppm = data.real_ppm.view(-1).to(mu_pred.device)
        is_carbon = data.is_carbon.view(-1).bool().to(mu_pred.device)
        mask = is_carbon & torch.isfinite(real_ppm) & (real_ppm > 0.0)
        if mask.any():
            all_mu_true.append(real_ppm[mask].detach().cpu())
            all_mu_pred.append(mu_pred.view(-1)[mask].detach().cpu())
    if not all_mu_true:
        return None, None, None, None
    mu_true = torch.cat(all_mu_true).numpy()
    mu_pred = torch.cat(all_mu_pred).numpy()
    pi_dummy = torch.ones_like(torch.tensor(mu_true)).numpy()
    return mu_true, mu_pred, pi_dummy, pi_dummy


def train(model, loader, opt, device, epoch=0, beta=1.0e-4):
    model.train()
    totals = {"loss": 0.0, "mu": 0.0, "kl": 0.0, "mae_ppm": 0.0}
    count = 0
    for data in loader:
        data = data.to(device)
        opt.zero_grad(set_to_none=True)
        pred, latent = model(data)
        loss, parts = g2s_loss_function(pred, latent, data, beta=beta)
        if torch.isnan(loss):
            continue
        loss.backward()
        opt.step()
        batch_size = int(data.real_ppm.numel())
        count += batch_size
        for key in totals:
            totals[key] += float(parts[key].detach().item()) * batch_size
    return tuple(totals[key] / max(count, 1) for key in ["loss", "mu", "kl"])


@torch.no_grad()
def validate(model, loader, device, beta=1.0e-4):
    model.eval()
    totals = {"loss": 0.0, "mu": 0.0, "kl": 0.0, "mae_ppm": 0.0}
    count = 0
    for data in loader:
        data = data.to(device)
        pred, latent = model(data)
        loss, parts = g2s_loss_function(pred, latent, data, beta=beta)
        if torch.isnan(loss):
            continue
        batch_size = int(data.real_ppm.numel())
        count += batch_size
        for key in totals:
            totals[key] += float(parts[key].detach().item()) * batch_size
    return (
        totals["loss"] / max(count, 1),
        totals["mu"] / max(count, 1),
        totals["kl"] / max(count, 1),
        totals["mae_ppm"] / max(count, 1),
        0.0,
    )


if __name__ == "__main__":
    import argparse
    from torch_geometric.loader import DataLoader

    parser = argparse.ArgumentParser("Train shift-only G2S model (legacy entry point)")
    parser.add_argument("data_path", type=str)
    parser.add_argument("--hid", type=int, default=384)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--k_hop", type=int, default=3)
    parser.add_argument("--beta", type=float, default=1.0e-4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    args = parser.parse_args()

    raw_list = load_raw_data(args.data_path)
    dataset = LocalNMRDataset(raw_list, k=args.k_hop)
    mol_ids = torch.unique(dataset._data.mol_id).tolist()
    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(mol_ids), generator=rng).tolist()
    mol_ids = [mol_ids[i] for i in perm]
    split = int(0.9 * len(mol_ids))
    train_mols = set(mol_ids[:split])
    val_mols = set(mol_ids[split:])
    train_idx = [i for i, d in enumerate(dataset) if int(d.mol_id.item()) in train_mols]
    val_idx = [i for i, d in enumerate(dataset) if int(d.mol_id.item()) in val_mols]
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = G2SConfig(k_hop=args.k_hop, hid=args.hid, latent_dim=args.latent_dim, beta=args.beta)
    model = NMR_VAE.from_config(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-5)
    best = float("inf")
    os.makedirs("checkpoints_g2s", exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_mu, train_kl = train(model, train_loader, opt, device, epoch, beta=args.beta)
        val_loss, _, _, val_mae, _ = validate(model, val_loader, device, beta=args.beta)
        print(f"Epoch {epoch:03d} | train {train_loss:.5f} | val {val_loss:.5f} | val MAE {val_mae:.3f} ppm")
        if val_loss < best:
            best = val_loss
            torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict(), "epoch": epoch}, "checkpoints_g2s/g2s_best_model.pt")
            print("  [best] saved checkpoints_g2s/g2s_best_model.pt")
