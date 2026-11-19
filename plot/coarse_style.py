"""Publication drawing primitives; graph coordinates never change graph connectivity."""

from dataclasses import asdict, dataclass

from matplotlib.colors import is_color_like, to_rgb
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np


COLORS = {
    "ink": "#173E68", "node": "#D7E6F3", "accent": "#E49A47",
    "accent_fill": "#FBE8D2", "context": "#9BAFBE", "context_fill": "#F0F5F8",
    "border": "#D6E1EB", "shell": "#F3F7FA", "center": "#D77169",
}
MPL_STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "lines.solid_capstyle": "round",
}


@dataclass(frozen=True)
class CoarseStyle:
    labeled_radius_pt: float = 7.0
    overview_radius_pt: float = 2.5
    font_size_pt: float = 7.0
    edge_width_pt: float = 1.5
    outline_width_pt: float = 0.8
    highlight_width_pt: float = 1.1
    motif_edge_length_mm: float = 6.0
    fragment_edge_length_mm: float = 4.5
    molecular_edge_length_mm: float = 10.5
    node_alpha: float = 0.9
    node_outline_color: str = "white"
    font_color: str = "white"
    edge_color: tuple[float, float, float] = (0.5, 0.5, 0.5)

    def __post_init__(self):
        dimensions = (value for name, value in asdict(self).items() if name.endswith(("_pt", "_mm")))
        if any(not np.isfinite(value) or value <= 0 for value in dimensions):
            raise ValueError("Coarse drawing dimensions must be finite and positive.")
        if not np.isfinite(self.node_alpha) or not 0 < self.node_alpha <= 1:
            raise ValueError("Node opacity must be in (0, 1].")
        if any(not is_color_like(color) for color in (self.node_outline_color, self.font_color, self.edge_color)):
            raise ValueError("Coarse drawing colors must be valid matplotlib colors.")


DEFAULT_STYLE = CoarseStyle()


def contrasting_label_color(fill):
    components = np.asarray(to_rgb(fill))
    linear = np.where(components <= 0.04045, components / 12.92, ((components + 0.055) / 1.055) ** 2.4)
    luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
    return "#111111" if (luminance + 0.05) / 0.05 >= 1.05 / (luminance + 0.05) else "white"


def model_su_colors(su_types):
    from model.shared.coarse_graph import _node_color

    return {int(node_id): _node_color(int(su_type)) for node_id, su_type in su_types.items()}


def normalize_positions(positions, edges, edge_length_mm=DEFAULT_STYLE.motif_edge_length_mm):
    coordinates = {int(node_id): np.asarray(position, dtype=float) for node_id, position in positions.items()}
    if not coordinates or any(position.shape != (2,) or not np.isfinite(position).all() for position in coordinates.values()):
        raise ValueError("A coarse graph requires finite two-dimensional positions.")
    lengths = [np.linalg.norm(coordinates[int(source)] - coordinates[int(target)]) for source, target in edges]
    if not lengths or min(lengths) <= 0 or not np.isfinite(edge_length_mm) or edge_length_mm <= 0:
        raise ValueError("Graph edges and target edge length must have positive lengths.")
    bounds = np.asarray(list(coordinates.values()))
    center = (bounds.max(axis=0) + bounds.min(axis=0)) / 2
    scale = edge_length_mm / float(np.median(lengths))
    return {node_id: (position - center) * scale for node_id, position in coordinates.items()}


def draw_graph(axis, graph, center=(0, 0), scale=1.0, *, labels=False, changed=(),
               required=False, node_colors=None, accent_edges=(), context_edges=(),
               accent_nodes=(), context_nodes=(), terminal_nodes=(), root_id=None,
               style=DEFAULT_STYLE):
    """Draw physical-size nodes and clipped edges, leaving the supplied graph untouched."""
    positions = {
        int(node_id): np.asarray(position, dtype=float) * scale + np.asarray(center)
        for node_id, position in graph["positions"].items()
    }
    if not positions or any(position.shape != (2,) or not np.isfinite(position).all() for position in positions.values()):
        raise ValueError("A coarse graph requires finite two-dimensional positions.")
    highlighted = set(map(int, changed))
    accent_nodes = set(map(int, accent_nodes))
    context_nodes = set(map(int, context_nodes))
    terminal_nodes = set(map(int, terminal_nodes))
    accent_edges = {frozenset(map(int, edge)) for edge in accent_edges}
    context_edges = {frozenset(map(int, edge)) for edge in context_edges}
    radius = style.labeled_radius_pt if labels else style.overview_radius_pt
    fills = {int(node_id): fill for node_id, fill in (node_colors or {}).items()}
    su_types = {int(node_id): int(su_type) for node_id, su_type in graph.get("su_types", {}).items()}
    if labels and not positions.keys() <= su_types.keys():
        raise ValueError("Every labeled node requires its actual SU type.")
    edge_artists, node_artists, text_artists = [], {}, {}
    for source, target in graph["edges"]:
        source, target = int(source), int(target)
        if source not in positions or target not in positions or source == target:
            raise ValueError("Edges must join two distinct displayed graph nodes.")
        resource = source in highlighted or target in highlighted
        key = frozenset((source, target))
        color = COLORS["accent"] if resource or key in accent_edges else COLORS["context"] if key in context_edges else style.edge_color
        edge = FancyArrowPatch(
            positions[source], positions[target], arrowstyle="-", shrinkA=radius, shrinkB=radius,
            linewidth=style.edge_width_pt, color=color,
            linestyle="--" if required and resource else "-", zorder=2,
        )
        axis.add_patch(edge)
        edge_artists.append(edge)
    for node_id, position in positions.items():
        resource = node_id in highlighted
        accent = node_id in accent_nodes
        context = node_id in context_nodes and not accent
        fill = fills.get(node_id, COLORS["accent_fill"] if accent else COLORS["context_fill"] if context else COLORS["node"])
        if node_id in terminal_nodes:
            fill = COLORS["accent"]
        if required and resource:
            fill = "white"
        outline = COLORS["accent"] if resource or accent else COLORS["context"] if context else style.node_outline_color
        node_artists[node_id] = axis.scatter(
            [position[0]], [position[1]], s=(2 * radius) ** 2, marker="o",
            facecolors=[fill], edgecolors=[outline],
            linewidths=style.highlight_width_pt if resource else style.outline_width_pt,
            linestyles=(0, (1.2, 1.1)) if required and resource else "-", alpha=style.node_alpha, zorder=4,
        )
        if labels:
            text_artists[node_id] = axis.text(
                *position, str(su_types[node_id]), ha="center", va="center",
                fontsize=style.font_size_pt, fontweight="bold",
                color=COLORS["accent"] if required and resource else style.font_color, zorder=5,
            )
        if node_id == root_id:
            axis.scatter(
                [position[0]], [position[1]], s=(2 * radius + 4) ** 2,
                facecolors="none", edgecolors=COLORS["center"], linewidths=0.9, zorder=3,
            )
    return {"positions": positions, "nodes": node_artists, "edges": edge_artists, "texts": text_artists, "radius_pt": radius}


def draw_neighborhood(axis, neighborhood, center, *, max_hop=2, labels=True, scale=1.0, style=DEFAULT_STYLE):
    if max_hop not in (1, 2):
        raise ValueError("The framework neighborhood supports one or two hops.")
    for radius in ([18.0, 9.2] if max_hop == 2 else [9.2]):
        axis.add_patch(Circle(center, radius * scale, facecolor=COLORS["shell"], edgecolor=COLORS["border"],
                              linewidth=0.6, linestyle="--", zorder=0))
    selected = {neighborhood["root_id"], *neighborhood["hop1_ids"]}
    if max_hop == 2:
        selected.update(neighborhood["hop2_ids"])
    original = neighborhood["graph"]
    graph = {
        "positions": {int(node_id): position for node_id, position in original["positions"].items() if int(node_id) in selected},
        "su_types": {int(node_id): su_type for node_id, su_type in original["su_types"].items() if int(node_id) in selected},
        "edges": [(source, target) for source, target in original["edges"] if source in selected and target in selected],
    }
    return draw_graph(axis, graph, center, scale, labels=labels, root_id=neighborhood["root_id"],
                      node_colors=model_su_colors(graph["su_types"]), style=style)
