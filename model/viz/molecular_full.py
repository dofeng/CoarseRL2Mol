from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math

smiles = "O=C(NC)C1=NC(SC)=C(C2=C1C=CC=C2)C3=CC(C(O)=O)=C(CC(O)C(OC)=O)C(C=C)=C3"

mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
AllChem.Compute2DCoords(mol)
conf = mol.GetConformer()

rotation_degrees = -60   # 顺时针90度；逆时针用 +90
label_rotation = 0

colors = {
    "C": "#05B9E2",
    "H": "#bdbdbd",
    "O": "#ff4040",
    "N": "#4040ff",
    "S": "#f2bf1a",
}

def rotate_positions(pos, degrees):
    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))

    theta = math.radians(degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    out = {}
    for idx, (x, y) in pos.items():
        dx = x - cx
        dy = y - cy
        out[idx] = (
            cx + dx * cos_t - dy * sin_t,
            cy + dx * sin_t + dy * cos_t,
        )
    return out

def draw_bond(ax, p1, p2, bond_order, color="#7f7f7f", lw=2.0):
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-8:
        return

    # 垂直方向单位向量，用于双键/三键平移
    nx = -dy / length
    ny = dx / length

    offset = 0.055

    if abs(bond_order - 1.0) < 1e-3:
        offsets = [0.0]
        linestyles = ["-"]
    elif abs(bond_order - 2.0) < 1e-3:
        offsets = [-offset, offset]
        linestyles = ["-", "-"]
    elif abs(bond_order - 3.0) < 1e-3:
        offsets = [-offset * 1.3, 0.0, offset * 1.3]
        linestyles = ["-", "-", "-"]
    elif abs(bond_order - 1.5) < 1e-3:
        # 芳香键：一条实线 + 一条虚线
        offsets = [-offset, offset]
        linestyles = ["-", (0, (3, 3))]
    else:
        offsets = [0.0]
        linestyles = ["-"]

    for off, ls in zip(offsets, linestyles):
        ax.plot(
            [x1 + nx * off, x2 + nx * off],
            [y1 + ny * off, y2 + ny * off],
            color=color,
            linewidth=lw,
            linestyle=ls,
            solid_capstyle="round",
            zorder=1,
        )

# 读取 RDKit 2D 坐标
pos = {}
for atom in mol.GetAtoms():
    p = conf.GetAtomPosition(atom.GetIdx())
    pos[atom.GetIdx()] = (p.x, p.y)

# 旋转整张结构
pos = rotate_positions(pos, rotation_degrees)

fig, ax = plt.subplots(figsize=(12, 16))

# 先画键，按键级画单/双/三/芳香
for bond in mol.GetBonds():
    i = bond.GetBeginAtomIdx()
    j = bond.GetEndAtomIdx()
    bond_order = bond.GetBondTypeAsDouble()
    draw_bond(ax, pos[i], pos[j], bond_order, lw=2.2)

# 再画原子节点
for atom in mol.GetAtoms():
    idx = atom.GetIdx()
    symbol = atom.GetSymbol()
    x, y = pos[idx]

    radius = 0.3 if symbol == "H" else 0.35

    circle = Circle(
        (x, y),
        radius,
        facecolor=colors.get(symbol, "#999999"),
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    ax.add_patch(circle)

    ax.text(
        x,
        y,
        symbol,
        ha="center",
        va="center",
        fontsize=18 if symbol == "H" else 18,
        color="black" if symbol == "H" else "white",
        fontweight="bold",
        fontfamily="Arial",
        rotation=label_rotation,
        rotation_mode="anchor",
        zorder=4,
    )

xs = [x for x, _ in pos.values()]
ys = [y for _, y in pos.values()]
margin = 0.8

ax.set_xlim(min(xs) - margin, max(xs) + margin)
ax.set_ylim(min(ys) - margin, max(ys) + margin)
ax.set_aspect("equal")
ax.axis("off")

plt.savefig(
    "full_atom_2d_node_molecule_rotated.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()