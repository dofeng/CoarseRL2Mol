import os
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Tuple
from .RL_state import AromaticCluster, SU_NAMES, count_graph_su_distribution, compute_su_delta

# Style parameters (same as mcts_visualization.py)
NODE_RADIUS = 0.18
NODE_ALPHA = 0.9
FONT_SIZE = 6
FONT_COLOR = 'white'
INTRA_EDGE_LW = 1.5
INTRA_EDGE_COLOR = (0.6, 0.6, 0.6)

def _node_color(su: int) -> Tuple[float, float, float]:
    """Node color by SU type (same as mcts_visualization.py)"""
    if su == 0:  return 0.55, 0.10, 0.10
    elif su == 1: return 0.80, 0.15, 0.15
    elif su == 2: return 0.85, 0.35, 0.15
    elif su == 3: return 0.75, 0.20, 0.35
    if su == 12:  return 0.9, 0.2, 0.2      # Bridgehead_C - red
    elif su == 10: return 0.2, 0.4, 0.9     # Aryl_Aro_C - blue
    elif su == 11: return 0.2, 0.8, 0.4     # Alkyl_Aro_C - green
    elif su == 13: return 0.4, 0.7, 0.9     # Aro_CH - light blue
    elif su == 14: return 0.78, 0.28, 0.62
    elif su == 15: return 0.95, 0.45, 0.55
    elif su == 16: return 0.95, 0.65, 0.35
    elif su == 17: return 0.45, 0.15, 0.75
    elif su == 18: return 0.65, 0.35, 0.90
    elif su == 5:  return 0.8, 0.3, 0.6     # O_Aro_C - pink
    elif su == 6:  return 0.6, 0.2, 0.8     # N_Aro_C - purple
    elif su == 7:  return 0.9, 0.8, 0.1     # S_Aro_C - gold
    elif su == 8:  return 0.5, 0.5, 0.5     # X_Aro_C - gray
    elif su == 9:  return 0.7, 0.1, 0.1     # Keto_Aro_C - dark red
    elif su in (23, 24, 25): return 0.9, 0.6, 0.2  # Alkyl
    elif su == 22: return 0.8, 0.5, 0.1     # CH3
    elif su == 19: return 0.9, 0.4, 0.4     # Alcohol_C
    elif su == 28: return 1.0, 0.2, 0.2     # OH
    elif su == 29: return 0.8, 0.0, 0.0     # Ether_O
    elif su == 27: return 0.5, 0.0, 0.8     # Amine_N
    elif su == 31: return 0.9, 0.9, 0.0     # Thioether_S
    elif su == 26: return 0.3, 0.0, 0.9     # Hetero_N
    elif su == 30: return 0.9, 0.7, 0.0     # Hetero_S
    else: return 0.6, 0.6, 0.6


def _build_uid_maps(graph, placed_clusters):
    uid_to_node = {}
    uid_to_pos = {}
    for c in placed_clusters:
        for s in c.sites:
            uid_to_node[s.uid] = s
            uid_to_pos[s.uid] = s.pos2d
    for cn in graph.chains:
        uid_to_node[cn.uid] = cn
        uid_to_pos[cn.uid] = cn.pos2d
    return uid_to_node, uid_to_pos


def _resolve_chain_nodes(chain, uid_to_node):
    resolved = []
    for node in chain:
        uid = getattr(node, 'uid', None)
        resolved.append(uid_to_node.get(uid, node))
    return resolved


def _legend_entries_for_su_types(su_types):
    from matplotlib.patches import Patch
    return [
        Patch(fc=_node_color(su), ec='k', label=f"{su}: {SU_NAMES.get(su, f'SU_{su}')}")
        for su in sorted(su_types)
    ]


def _rotate_xy(x: float, y: float, angle_rad: float) -> Tuple[float, float]:
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    return (x * ca - y * sa, x * sa + y * ca)


def _compute_display_transform(positions: List[Tuple[float, float]]) -> Tuple[float, float, float, Tuple[float, float, float, float]]:
    if not positions:
        return 0.0, 0.0, 0.0, (0.0, 0.0, 1.0, 1.0)

    cx = sum(x for x, _ in positions) / len(positions)
    cy = sum(y for _, y in positions) / len(positions)
    centered = [(x - cx, y - cy) for x, y in positions]
    xs = [p[0] for p in centered]
    ys = [p[1] for p in centered]
    bbox = (min(xs), max(xs), min(ys), max(ys))
    # Keep the original molecular orientation. We only recenter and resize
    # the canvas now; no automatic rotation.
    return cx, cy, 0.0, bbox


def _transform_pos(pos: Tuple[float, float], cx: float, cy: float, angle_rad: float) -> Tuple[float, float]:
    return _rotate_xy(pos[0] - cx, pos[1] - cy, angle_rad)


def _figure_size_from_bbox(bbox: Tuple[float, float, float, float], legend: bool = True) -> Tuple[float, float]:
    min_x, max_x, min_y, max_y = bbox
    w = max(max_x - min_x, 1.0)
    h = max(max_y - min_y, 1.0)
    plot_h = 12.0
    plot_w = min(22.0, max(10.0, plot_h * (w / h)))
    total_w = plot_w + (4.8 if legend else 0.0)
    return total_w, plot_h


def _place_legend_outside(ax, legend_elements, fontsize: int = 8):
    fig = ax.figure
    fig.subplots_adjust(left=0.04, right=0.78, top=0.93, bottom=0.06)
    return ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=fontsize,
        framealpha=0.95,
    )


def draw_cluster(ax, cluster: AromaticCluster, offset: Tuple[float, float] = (0, 0)):
    """Draw single aromatic cluster with mcts_visualization style"""
    ox, oy = offset
    
    # Draw intra-ring edges (gray)
    for i, j in cluster.edges:
        if i < len(cluster.sites) and j < len(cluster.sites):
            s1, s2 = cluster.sites[i], cluster.sites[j]
            ax.plot([s1.pos2d[0] - ox, s2.pos2d[0] - ox],
                    [s1.pos2d[1] - oy, s2.pos2d[1] - oy],
                    color=INTRA_EDGE_COLOR, linewidth=INTRA_EDGE_LW, zorder=1)
    
    # Draw site nodes
    for site in cluster.sites:
        color = _node_color(site.su_type)
        x, y = site.pos2d[0] - ox, site.pos2d[1] - oy
        ax.add_patch(Circle((x, y), NODE_RADIUS, color=color, alpha=NODE_ALPHA, zorder=3))
        ax.text(x, y, f"{site.su_type}", fontsize=FONT_SIZE, ha='center', va='center',
                color=FONT_COLOR, zorder=4)


def visualize_all_rigid_clusters(rigid_cluster_copies: List[dict], output_path: str, 
                                   title: str = "All Rigid Clusters"):
    """
    Visualize ALL rigid clusters (connected + individual) in a single unified figure.
    """
    if not rigid_cluster_copies:
        print("[Visualization] No rigid clusters to visualize")
        return
    
    n_clusters = len(rigid_cluster_copies)
    n_cols = min(4, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    
    # Style constants
    RIGID_EDGE_COLOR = (0.9, 0.1, 0.1)
    RIGID_EDGE_LW = 2.0
    
    for idx, rc_data in enumerate(rigid_cluster_copies):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        aromatic_clusters = rc_data.get('aromatic_clusters', [])
        rigid_edges = rc_data.get('rigid_edges', [])
        
        # Draw aromatic clusters (already coordinate-normalized)
        for cluster in aromatic_clusters:
            draw_cluster(ax, cluster)
        
        # Draw rigid edges
        for edge in rigid_edges:
            if hasattr(edge, 'line') and edge.line and len(edge.line) >= 2:
                # Need to recalculate edge line in local coords
                # Find sites by uid
                x1, y1 = None, None
                x2, y2 = None, None
                for c in aromatic_clusters:
                    for s in c.sites:
                        if s.uid == edge.u:
                            x1, y1 = s.pos2d
                        if s.uid == edge.v:
                            x2, y2 = s.pos2d
                if x1 is not None and x2 is not None:
                    ax.plot([x1, x2], [y1, y2], color=RIGID_EDGE_COLOR, 
                            linewidth=RIGID_EDGE_LW, zorder=2, alpha=0.8)
        
        # Adjust view based on cluster positions
        if aromatic_clusters:
            all_x = [s.pos2d[0] for c in aromatic_clusters for s in c.sites]
            all_y = [s.pos2d[1] for c in aromatic_clusters for s in c.sites]
            if all_x and all_y:
                margin = 1.0
                ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
                ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Title: NEW ID, size, composition, type
        rc_id = rc_data.get('id', idx)
        rc_size = rc_data.get('size', 0)
        rc_composition = rc_data.get('composition', 'unknown')
        rc_type = rc_data.get('type', 'unknown')
        
        type_label = '(Connected)' if rc_type == 'connected' else '(Individual)'
        ax.set_title(f"Rigid Cluster #{rc_id}: {rc_size} rings\n{rc_composition} {type_label}",
                    fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_clusters, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    # Count types for title
    n_connected = sum(1 for rc in rigid_cluster_copies if rc.get('type') == 'connected')
    n_individual = sum(1 for rc in rigid_cluster_copies if rc.get('type') == 'individual')
    
    plt.suptitle(f"{title}\n{n_clusters} total: {n_connected} connected, {n_individual} individual",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"[Visualization] Saved all {n_clusters} rigid clusters to {output_path}")


def save_flex_result(state, output_path: str, title: str = "Flex Connection Result"):
    """
    Save flex stage result visualization showing clusters + rigid edges + flex chains.
    
    Args:
        state: MCTSState after flex stage
        output_path: Output file path
        title: Plot title
    """
    from .RL_state import HexGrid, ChainNode
    
    graph = state.graph
    clusters = graph.clusters
    placed_clusters = [c for c in clusters if c.placed]
    uid_to_node, uid_to_pos = _build_uid_maps(graph, placed_clusters)
    
    if not placed_clusters:
        return
    
    # Style constants
    RIGID_EDGE_COLOR = (0.1, 0.2, 0.8)
    RIGID_EDGE_LW = 2.0
    FLEX_EDGE_COLOR = (0.1, 0.7, 0.3)
    FLEX_EDGE_LW = 2.4
    
    # Collect all positions for centering
    all_positions = []
    for c in placed_clusters:
        for s in c.sites:
            all_positions.append(s.pos2d)
    for cn in graph.chains:
        all_positions.append(cn.pos2d)
    
    if not all_positions:
        return
    
    ref_x, ref_y, angle_rad, display_bbox = _compute_display_transform(all_positions)
    fig, ax = plt.subplots(figsize=_figure_size_from_bbox(display_bbox, legend=True))
    tpos = lambda pos: _transform_pos(pos, ref_x, ref_y, angle_rad)
    
    # 1. Draw intra-ring edges (gray)
    for cluster in placed_clusters:
        for i, j in cluster.edges:
            if i < len(cluster.sites) and j < len(cluster.sites):
                s1, s2 = cluster.sites[i], cluster.sites[j]
                p1 = tpos(s1.pos2d)
                p2 = tpos(s2.pos2d)
                ax.plot([p1[0], p2[0]],
                        [p1[1], p2[1]],
                        color=INTRA_EDGE_COLOR, linewidth=INTRA_EDGE_LW, zorder=1)
    
    # 2. Draw rigid edges (blue)
    for edge in graph.rigid:
        x1, y1, x2, y2 = None, None, None, None
        for c in placed_clusters:
            for s in c.sites:
                if s.uid == edge.u:
                    x1, y1 = s.pos2d
                if s.uid == edge.v:
                    x2, y2 = s.pos2d
        if x1 is not None and x2 is not None:
            p1 = tpos((x1, y1))
            p2 = tpos((x2, y2))
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=RIGID_EDGE_COLOR, linewidth=RIGID_EDGE_LW, zorder=2)
    
    # 3. Draw flex edges (green) with chain nodes
    for edge in graph.flex:
        chain = _resolve_chain_nodes(edge.chain, uid_to_node)
        if not chain:
            continue
        
        # Find source site
        src_raw = uid_to_pos.get(edge.u)
        src_pos = tpos(src_raw) if src_raw else None
        
        # Draw source -> first chain node
        if src_pos and chain:
            first_cn = chain[0]
            first_pos = tpos(first_cn.pos2d)
            ax.plot([src_pos[0], first_pos[0]],
                    [src_pos[1], first_pos[1]],
                    color=FLEX_EDGE_COLOR, linewidth=FLEX_EDGE_LW, zorder=2)
        
        # Draw chain node links
        for i in range(len(chain) - 1):
            cn1, cn2 = chain[i], chain[i+1]
            p1 = tpos(cn1.pos2d)
            p2 = tpos(cn2.pos2d)
            ax.plot([p1[0], p2[0]],
                    [p1[1], p2[1]],
                    color=FLEX_EDGE_COLOR, linewidth=FLEX_EDGE_LW, zorder=2)
        
        # Draw last chain node -> target site
        tgt_raw = uid_to_pos.get(edge.v)
        tgt_pos = tpos(tgt_raw) if tgt_raw else None
        if tgt_pos and chain:
            last_cn = chain[-1]
            last_pos = tpos(last_cn.pos2d)
            ax.plot([last_pos[0], tgt_pos[0]],
                    [last_pos[1], tgt_pos[1]],
                    color=FLEX_EDGE_COLOR, linewidth=FLEX_EDGE_LW, zorder=2)
        
        # Draw chain nodes
        for cn in chain:
            color = _node_color(cn.su_type)
            x, y = tpos(cn.pos2d)
            ax.add_patch(Circle((x, y), NODE_RADIUS * 0.9, color=color, alpha=NODE_ALPHA, zorder=3))
            ax.text(x, y, f"{cn.su_type}", fontsize=FONT_SIZE, ha='center', va='center',
                    color=FONT_COLOR, zorder=4)
    
    # 4. Draw cluster nodes (on top)
    for cluster in placed_clusters:
        for site in cluster.sites:
            color = _node_color(site.su_type)
            x, y = tpos(site.pos2d)
            ax.add_patch(Circle((x, y), NODE_RADIUS, color=color, alpha=NODE_ALPHA, zorder=5))
            ax.text(x, y, f"{site.su_type}", fontsize=FONT_SIZE, ha='center', va='center',
                    color=FONT_COLOR, zorder=6)
    
    min_x, max_x, min_y, max_y = display_bbox
    margin = 1.0
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Stats text
    n_flex = len(graph.flex)
    n_rigid = len(graph.rigid)
    n_chains = sum(len(e.chain) for e in graph.flex)
    n_comp = state.get_component_count()
    stats = (f"Clusters: {len(placed_clusters)}, Components: {n_comp}\n"
             f"Rigid edges: {n_rigid}, Flex edges: {n_flex}, Chain nodes: {n_chains}")
    ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    
    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    present_su = {s.su_type for c in placed_clusters for s in c.sites}
    present_su.update(cn.su_type for cn in graph.chains)
    legend_elements = _legend_entries_for_su_types(present_su)
    legend_elements.extend([
        Line2D([0], [0], color=RIGID_EDGE_COLOR, linewidth=RIGID_EDGE_LW, label='Rigid Edge'),
        Line2D([0], [0], color=FLEX_EDGE_COLOR, linewidth=FLEX_EDGE_LW, label='Flex Edge'),
    ])
    legend = _place_legend_outside(ax, legend_elements, fontsize=8)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', bbox_extra_artists=(legend,))
    plt.close(fig)


def save_side_result(state, output_path: str, title: str = "Side Chain Result"):
    """
    Save side stage result visualization showing clusters + rigid + flex + side chains.
    
    Args:
        state: MCTSState after side stage
        output_path: Output file path
        title: Plot title
    """
    from .RL_state import HexGrid, ChainNode
    
    graph = state.graph
    clusters = graph.clusters
    placed_clusters = [c for c in clusters if c.placed]
    uid_to_node, uid_to_pos = _build_uid_maps(graph, placed_clusters)
    
    if not placed_clusters:
        return
    
    # Style constants
    RIGID_EDGE_COLOR = (0.1, 0.2, 0.8)
    RIGID_EDGE_LW = 2.0
    FLEX_EDGE_COLOR = (0.1, 0.7, 0.3)
    FLEX_EDGE_LW = 2.4
    SIDE_EDGE_COLOR = (0.9, 0.5, 0.1)    # Orange for side chains
    SIDE_EDGE_LW = 1.8
    BRANCH_EDGE_COLOR = (0.6, 0.2, 0.8)  # Purple for branches
    BRANCH_EDGE_LW = 1.6
    
    # Collect all positions for centering
    all_positions = []
    for c in placed_clusters:
        for s in c.sites:
            all_positions.append(s.pos2d)
    for cn in graph.chains:
        all_positions.append(cn.pos2d)
    
    if not all_positions:
        return
    
    ref_x, ref_y, angle_rad, display_bbox = _compute_display_transform(all_positions)
    fig, ax = plt.subplots(figsize=_figure_size_from_bbox(display_bbox, legend=True))
    tpos = lambda pos: _transform_pos(pos, ref_x, ref_y, angle_rad)
    
    # 1. Draw intra-ring edges (gray)
    for cluster in placed_clusters:
        for i, j in cluster.edges:
            if i < len(cluster.sites) and j < len(cluster.sites):
                s1, s2 = cluster.sites[i], cluster.sites[j]
                p1 = tpos(s1.pos2d)
                p2 = tpos(s2.pos2d)
                ax.plot([p1[0], p2[0]],
                        [p1[1], p2[1]],
                        color=INTRA_EDGE_COLOR, linewidth=INTRA_EDGE_LW, zorder=1)
    
    # 2. Draw rigid edges (blue)
    for edge in graph.rigid:
        x1, y1 = uid_to_pos.get(edge.u, (None, None))
        x2, y2 = uid_to_pos.get(edge.v, (None, None))
        if x1 is not None and x2 is not None:
            p1 = tpos((x1, y1))
            p2 = tpos((x2, y2))
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=RIGID_EDGE_COLOR, linewidth=RIGID_EDGE_LW, zorder=2)
    
    # 3. Draw flex edges (green) with chain nodes
    for edge in graph.flex:
        chain = _resolve_chain_nodes(edge.chain, uid_to_node)
        if not chain:
            continue
        src_pos = uid_to_pos.get(edge.u)
        if src_pos and chain:
            p_src = tpos(src_pos)
            p_first = tpos(chain[0].pos2d)
            ax.plot([p_src[0], p_first[0]],
                    [p_src[1], p_first[1]],
                    color=FLEX_EDGE_COLOR, linewidth=FLEX_EDGE_LW, zorder=2)
        for i in range(len(chain) - 1):
            cn1, cn2 = chain[i], chain[i+1]
            p1 = tpos(cn1.pos2d)
            p2 = tpos(cn2.pos2d)
            ax.plot([p1[0], p2[0]],
                    [p1[1], p2[1]],
                    color=FLEX_EDGE_COLOR, linewidth=FLEX_EDGE_LW, zorder=2)
        tgt_pos = uid_to_pos.get(edge.v)
        if tgt_pos and chain:
            p_last = tpos(chain[-1].pos2d)
            p_tgt = tpos(tgt_pos)
            ax.plot([p_last[0], p_tgt[0]],
                    [p_last[1], p_tgt[1]],
                    color=FLEX_EDGE_COLOR, linewidth=FLEX_EDGE_LW, zorder=2)
        # chain node circles
        for cn in chain:
            color = _node_color(cn.su_type)
            x, y = tpos(cn.pos2d)
            ax.add_patch(Circle((x, y), NODE_RADIUS * 0.9, color=color, alpha=NODE_ALPHA, zorder=3))
            ax.text(x, y, f"{cn.su_type}", fontsize=FONT_SIZE, ha='center', va='center',
                    color=FONT_COLOR, zorder=4)
    
    # 4. Draw side chain edges (orange) + nodes
    for edge in graph.side:
        chain = _resolve_chain_nodes(edge.chain, uid_to_node)
        if not chain:
            continue
        # source site → first chain node
        src_pos = uid_to_pos.get(edge.u)
        if src_pos and chain:
            p_src = tpos(src_pos)
            p_first = tpos(chain[0].pos2d)
            ax.plot([p_src[0], p_first[0]],
                    [p_src[1], p_first[1]],
                    color=SIDE_EDGE_COLOR, linewidth=SIDE_EDGE_LW, zorder=2)
        # inter-chain links
        for i in range(len(chain) - 1):
            cn1, cn2 = chain[i], chain[i+1]
            p1 = tpos(cn1.pos2d)
            p2 = tpos(cn2.pos2d)
            ax.plot([p1[0], p2[0]],
                    [p1[1], p2[1]],
                    color=SIDE_EDGE_COLOR, linewidth=SIDE_EDGE_LW, zorder=2)
        # chain node circles
        for cn in chain:
            color = _node_color(cn.su_type)
            x, y = tpos(cn.pos2d)
            ax.add_patch(Circle((x, y), NODE_RADIUS * 0.85, color=color, alpha=NODE_ALPHA, zorder=3))
            ax.text(x, y, f"{cn.su_type}", fontsize=FONT_SIZE, ha='center', va='center',
                    color=FONT_COLOR, zorder=4)
    
    # 5. Draw branch edges (purple) + nodes
    for edge in graph.branch:
        chain = _resolve_chain_nodes(edge.chain, uid_to_node)
        if not chain:
            continue
        # base can be a chain node uid — search chains for it
        base_pos = uid_to_pos.get(edge.base)
        if base_pos and chain:
            p_base = tpos(base_pos)
            p_first = tpos(chain[0].pos2d)
            ax.plot([p_base[0], p_first[0]],
                    [p_base[1], p_first[1]],
                    color=BRANCH_EDGE_COLOR, linewidth=BRANCH_EDGE_LW, zorder=2)
        for i in range(len(chain) - 1):
            cn1, cn2 = chain[i], chain[i+1]
            p1 = tpos(cn1.pos2d)
            p2 = tpos(cn2.pos2d)
            ax.plot([p1[0], p2[0]],
                    [p1[1], p2[1]],
                    color=BRANCH_EDGE_COLOR, linewidth=BRANCH_EDGE_LW, zorder=2)
        # Draw chain[-1] -> target (for closed rings like side rings)
        if edge.target and chain:
            target_pos = uid_to_pos.get(edge.target)
            if target_pos:
                last_cn = chain[-1]
                p_last = tpos(last_cn.pos2d)
                p_tgt = tpos(target_pos)
                ax.plot([p_last[0], p_tgt[0]],
                        [p_last[1], p_tgt[1]],
                        color=BRANCH_EDGE_COLOR, linewidth=BRANCH_EDGE_LW, zorder=2)
        for cn in chain:
            color = _node_color(cn.su_type)
            x, y = tpos(cn.pos2d)
            ax.add_patch(Circle((x, y), NODE_RADIUS * 0.85, color=color, alpha=NODE_ALPHA, zorder=3))
            ax.text(x, y, f"{cn.su_type}", fontsize=FONT_SIZE, ha='center', va='center',
                    color=FONT_COLOR, zorder=4)
    
    # 6. Draw cluster site nodes (on top)
    for cluster in placed_clusters:
        for site in cluster.sites:
            color = _node_color(site.su_type)
            x, y = tpos(site.pos2d)
            ax.add_patch(Circle((x, y), NODE_RADIUS, color=color, alpha=NODE_ALPHA, zorder=5))
            ax.text(x, y, f"{site.su_type}", fontsize=FONT_SIZE, ha='center', va='center',
                    color=FONT_COLOR, zorder=6)
    
    min_x, max_x, min_y, max_y = display_bbox
    margin = 1.0
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Stats
    n_flex = len(graph.flex)
    n_rigid = len(graph.rigid)
    n_side = len(graph.side)
    n_branch = len(graph.branch)
    n_side_nodes = sum(len(e.chain) for e in graph.side)
    n_comp = state.get_component_count()
    stats = (f"Clusters: {len(placed_clusters)}, Components: {n_comp}\n"
             f"Rigid: {n_rigid}, Flex: {n_flex}, "
             f"Side chains: {n_side} ({n_side_nodes} nodes), Branches: {n_branch}")
    ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    
    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    present_su = {s.su_type for c in placed_clusters for s in c.sites}
    present_su.update(cn.su_type for cn in graph.chains)
    legend_elements = _legend_entries_for_su_types(present_su)
    legend_elements.extend([
        Line2D([0], [0], color=RIGID_EDGE_COLOR, linewidth=RIGID_EDGE_LW, label='Rigid'),
        Line2D([0], [0], color=FLEX_EDGE_COLOR, linewidth=FLEX_EDGE_LW, label='Flex'),
        Line2D([0], [0], color=SIDE_EDGE_COLOR, linewidth=SIDE_EDGE_LW, label='Side chain'),
        Line2D([0], [0], color=BRANCH_EDGE_COLOR, linewidth=BRANCH_EDGE_LW, label='Branch'),
    ])
    legend = _place_legend_outside(ax, legend_elements, fontsize=8)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', bbox_extra_artists=(legend,))
    plt.close(fig)


def save_side_beam_results(candidates, summaries, output_dir: str):
    """
    Save visualization for multiple beam candidates from side stage.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_candidates = len(candidates)
    if n_candidates == 0:
        return
    
    for idx, (cand, summary) in enumerate(zip(candidates, summaries)):
        cand_dir = os.path.join(output_dir, f"candidate_{idx+1}")
        os.makedirs(cand_dir, exist_ok=True)
        
        # Save side result visualization
        side_path = os.path.join(cand_dir, "side_result.png")
        save_side_result(
            cand.state, side_path,
            title=f"Candidate #{idx+1} - Side Chains (score={cand.score:.3f})"
        )
        
        # Save text info
        info_path = os.path.join(cand_dir, "side_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"Candidate #{idx+1}\n")
            f.write(f"Score: {cand.score:.4f}\n")
            f.write(f"Stage: side\n\n")
            
            if isinstance(summary, dict):
                f.write(f"Side chains placed: {summary.get('sides_placed', 0)}/{summary.get('sides_total', 0)}\n")
                f.write(f"Side edges: {summary.get('side_edges', 0)}\n")
                f.write(f"Side chain nodes: {summary.get('side_chain_nodes', 0)}\n")
            
            # Detail
            f.write(f"\nSide Edge Details:\n")
            f.write("-" * 50 + "\n")
            state = cand.state
            chain_lookup = {cn.uid: cn for cn in state.graph.chains}
            for ei, edge in enumerate(state.graph.side):
                chain_sus = [chain_lookup.get(cn.uid, cn).su_type for cn in edge.chain]
                f.write(f"  Side {ei+1}: site={edge.u}, chain_len={len(edge.chain)}, SUs={chain_sus}\n")
    
    print(f"[Visualization] Saved {n_candidates} side candidate visualizations to {output_dir}")


def save_flex_beam_results(candidates, summaries, output_dir: str):
    """
    Save visualization for multiple beam candidates from flex stage.
    Each candidate gets:
      - flex_result.png: Full skeleton with rigid + flex connections
      - flex_info.txt: Text summary of flex stage results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_candidates = len(candidates)
    if n_candidates == 0:
        return
    
    for idx, (cand, summary) in enumerate(zip(candidates, summaries)):
        cand_dir = os.path.join(output_dir, f"candidate_{idx+1}")
        os.makedirs(cand_dir, exist_ok=True)
        
        # Save flex result visualization
        flex_path = os.path.join(cand_dir, "flex_result.png")
        save_flex_result(
            cand.state, flex_path,
            title=f"Candidate #{idx+1} - Flex Connections (score={cand.score:.3f})"
        )
        
        # Save text info
        info_path = os.path.join(cand_dir, "flex_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"Candidate #{idx+1}\n")
            f.write(f"Score: {cand.score:.4f}\n")
            f.write(f"Stage: flex\n\n")
            
            if isinstance(summary, dict):
                f.write(f"Flex edges: {summary.get('flex_edges', 0)}\n")
                f.write(f"Connections: {summary.get('connections_made', 0)}/{summary.get('connections_needed', 0)}\n")
                f.write(f"All connected: {summary.get('all_connected', False)}\n")
                f.write(f"Components: {summary.get('components', '?')}\n")
                f.write(f"sp2 ratio: {summary.get('sp2_ratio', 0):.2%}\n")
                f.write(f"Quota tier: {summary.get('quota_tier', 0)}\n")
                f.write(f"SU 23 consumed/quota: {summary.get('23_consumed', 0)}/{summary.get('23_quota', 0)}\n")
                f.write(f"SU 11 consumed: {summary.get('11_consumed', 0)}\n")
                f.write(f"H chains: {summary.get('h_chains', 0)}\n")
                f.write(f"V chains: {summary.get('v_chains', 0)}\n")
                f.write(f"Aspect ratio: {summary.get('aspect_ratio', 0):.2f}\n")
            
            # List flex edges detail
            f.write(f"\nFlex Edge Details:\n")
            f.write("-" * 50 + "\n")
            state = cand.state
            chain_lookup = {cn.uid: cn for cn in state.graph.chains}
            for ei, edge in enumerate(state.graph.flex):
                chain_len = len(edge.chain)
                chain_sus = [chain_lookup.get(cn.uid, cn).su_type for cn in edge.chain]
                f.write(f"  Edge {ei+1}: {edge.u} -> {edge.v}, chain_len={chain_len}, SUs={chain_sus}\n")
    
    print(f"[Visualization] Saved {n_candidates} flex candidate visualizations to {output_dir}")


def save_rigid_beam_results(candidates, summaries, output_dir: str):
    """
    Save visualization for multiple beam candidates from rigid stage.
    Only outputs all_rigid_clusters.png and rigid_clusters.txt per candidate.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_candidates = len(candidates)
    if n_candidates == 0:
        return
    
    from .stage_rigid import RigidStage
    
    for idx, (cand, summary) in enumerate(zip(candidates, summaries)):
        cand_dir = os.path.join(output_dir, f"candidate_{idx+1}")
        os.makedirs(cand_dir, exist_ok=True)
        
        # Save unified ALL rigid clusters view
        stage = RigidStage(cand.state, skip_init=True)
        rigid_copies = stage.create_rigid_cluster_copies()
        if rigid_copies:
            all_rc_path = os.path.join(cand_dir, "all_rigid_clusters.png")
            visualize_all_rigid_clusters(
                rigid_copies, 
                all_rc_path,
                title=f"Candidate #{idx+1} - All Rigid Clusters ({len(rigid_copies)} total)"
            )
        
        # Save rigid cluster info text
        info_path = os.path.join(cand_dir, "rigid_clusters.txt")
        with open(info_path, 'w') as f:
            f.write(f"Candidate #{idx+1}\n")
            f.write(f"Score: {cand.score:.4f}\n")
            f.write(f"Rigid edges: {summary['rigid_edges']}\n")
            f.write(f"Total clusters: {summary['total_clusters']} (all placed)\n")
            f.write(f"Remaining SU 10: {summary.get('remaining_su10', '?')}\n")
            f.write(f"\nRigid Clusters for Next Stage ({summary.get('rigid_cluster_count', 0)} total):\n")
            f.write("-" * 50 + "\n")
            
            for rc in summary.get('rigid_clusters', []):
                label = "10-10" if rc.get('has_rigid_edges') else "standalone"
                kinds_str = '+'.join(rc['kinds'])
                f.write(f"RC#{rc['id']}: {rc['total_rings']} rings, "
                        f"{rc['num_clusters']} clusters ({label}) [{kinds_str}]\n")
                f.write(f"  Member IDs: {rc['member_ids']}\n")
    
    print(f"[Visualization] Saved {n_candidates} candidate visualizations to {output_dir}")


def save_subst_beam_results(candidates, summaries, output_dir: str):
    """
    Save visualization for multiple beam candidates from substitution stage.
    We reuse the side result plotting since the graph structure is complete.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    n_candidates = len(candidates)
    if n_candidates == 0:
        return
        
    for idx, (cand, summary) in enumerate(zip(candidates, summaries)):
        cand_dir = os.path.join(output_dir, f"candidate_{idx+1}")
        os.makedirs(cand_dir, exist_ok=True)
        summary_payload = dict(summary or {})
        subst_summary = dict(summary_payload.get('subst_result', {}) or {})
        merged_summary = dict(summary_payload)
        merged_summary.update(subst_summary)
        
        # Plot final graph
        subst_path = os.path.join(cand_dir, "subst_result.png")
        save_side_result(
            cand.state, subst_path,
            title=f"Candidate #{idx+1} - Substitution (score={cand.score:.3f}, NMR={merged_summary.get('nmr_score', cand.info.get('nmr_score', 0)):.3f})"
        )
        
        # Save text info
        info_path = os.path.join(cand_dir, "subst_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"Candidate #{idx+1}\n")
            f.write(f"Score: {cand.score:.4f}\n")
            f.write(f"NMR Score: {cand.info.get('nmr_score', merged_summary.get('nmr_score', 0)):.4f}\n")
            f.write(f"Cosine Sim: {cand.info.get('nmr_cos_sim', merged_summary.get('nmr_cos_sim', 0)):.4f}\n")
            f.write(f"Stage: substitution\n\n")
            
            f.write(f"Seed: {merged_summary.get('seed', 'N/A')}\n")
            f.write(f"Subst Success: {merged_summary.get('success', cand.info.get('subst_success', False))}\n")
            f.write(f"Match Target: {merged_summary.get('complete', cand.info.get('subst_complete', False))}\n")
            f.write(f"L1 Delta: {merged_summary.get('l1_delta', cand.info.get('subst_l1_delta', 0))}\n\n")
            
            # Print SUs info
            state = cand.state
            counts = dict(
                merged_summary.get('after_distribution', {})
                or count_graph_su_distribution(
                    state.graph,
                    dedupe_chain_uids=True,
                    dedupe_axials=False,
                )
            )
            before_counts = dict(merged_summary.get('before_distribution', {}) or {})
            target_counts = dict(merged_summary.get('target_distribution', {}) or {})
            change_delta = dict(merged_summary.get('applied', {}) or compute_su_delta(counts, before_counts))
            target_delta = dict(merged_summary.get('remaining', {}) or compute_su_delta(counts, target_counts))
                
            f.write("Final SU Distribution:\n")
            f.write("-" * 50 + "\n")
            for su, cnt in sorted(counts.items()):
                f.write(f"  SU {su}: {cnt}\n")

            f.write("\nTarget SU Distribution:\n")
            f.write("-" * 50 + "\n")
            if target_counts:
                for su, cnt in sorted(target_counts.items()):
                    if int(cnt) > 0:
                        f.write(f"  SU {su}: {cnt}\n")
            else:
                f.write("  none\n")

            f.write("\nDelta (after-before):\n")
            f.write("-" * 50 + "\n")
            if change_delta:
                for su, cnt in sorted(change_delta.items()):
                    f.write(f"  SU {su}: {cnt:+d}\n")
            else:
                f.write("  none\n")

            f.write("\nDelta (actual-target):\n")
            f.write("-" * 50 + "\n")
            if target_delta:
                for su, cnt in sorted(target_delta.items()):
                    f.write(f"  SU {su}: {cnt:+d}\n")
            else:
                f.write("  none\n")

    print(f"[Visualization] Saved {n_candidates} subst candidate visualizations to {output_dir}")

def save_branch_beam_results(candidates, summaries, output_dir: str):
    """
    Save visualization for multiple beam candidates from branch stage.
    Each candidate gets:
      - branch_result.png: Full skeleton with rigid + flex + side + branch
      - branch_info.txt: Text summary of branch stage results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_candidates = len(candidates)
    if n_candidates == 0:
        return
    
    for idx, (cand, summary) in enumerate(zip(candidates, summaries)):
        cand_dir = os.path.join(output_dir, f"candidate_{idx+1}")
        os.makedirs(cand_dir, exist_ok=True)
        
        # Reuse save_side_result which already draws branch edges
        branch_path = os.path.join(cand_dir, "branch_result.png")
        save_side_result(
            cand.state, branch_path,
            title=f"Candidate #{idx+1} - Branch Stage (score={cand.score:.3f})"
        )
        
        # Save text info
        info_path = os.path.join(cand_dir, "branch_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"Candidate #{idx+1}\n")
            f.write(f"Score: {cand.score:.4f}\n")
            f.write(f"Stage: branch\n\n")
            
            if isinstance(summary, dict):
                f.write(f"Branches placed: {summary.get('branches_placed', 0)}/{summary.get('branches_total', 0)}\n")
                f.write(f"Branch edges: {summary.get('branch_edges', 0)}\n")
                f.write(f"Branch chain nodes: {summary.get('branch_nodes', 0)}\n")
            
            # Detail
            f.write(f"\nBranch Edge Details:\n")
            f.write("-" * 50 + "\n")
            state = cand.state
            chain_lookup = {cn.uid: cn for cn in state.graph.chains}
            for ei, edge in enumerate(state.graph.branch):
                chain_sus = [chain_lookup.get(cn.uid, cn).su_type for cn in edge.chain]
                target_str = f" -> {edge.target}" if edge.target else ""
                f.write(f"  Branch {ei+1}: base={edge.base}{target_str}, "
                        f"chain_len={len(edge.chain)}, SUs={chain_sus}\n")
    
    print(f"[Visualization] Saved {n_candidates} branch candidate visualizations to {output_dir}")
