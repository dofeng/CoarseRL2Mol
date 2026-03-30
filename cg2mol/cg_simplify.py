#!/usr/bin/env python3
"""
cg_simplify.py - 将MCTS输出的粗粒度分子JSON简化为节点+连接格式

输入：MCTS搜索输出的完整状态JSON（包含clusters, edges等）
输出：简化的粗粒度分子描述（类似PDB格式）

输出格式：
{
    "nodes": [
        {"id": "C0-V0", "seq": 0, "su_type": 7, "axial_coord": [-7, -4], "xy_coord": [-6.06, -0.50]},
        {"id": "C0-V1", "seq": 1, "su_type": 9, "axial_coord": [-7, -3], "xy_coord": [-6.06,  0.50]},
        ...
    ],
    "bonds": [
        [0, 1],   # 节点序号对
        [1, 3],
        .
    ],
    "metadata": {
        "source": "原始文件路径",
        "node_count": 节点总数,
        "bond_count": 连接总数,
        "created_at": "时间戳"
    }
}
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any


def simplify_cg_molecule(mcts_json: Dict) -> Dict:
    """
    将MCTS输出的粗粒度分子JSON简化为节点+连接格式
    
    Args:
        mcts_json: MCTS搜索输出的完整状态JSON
    
    Returns:
        简化后的JSON字典
    """
    # 节点列表和ID到序号的映射
    nodes: List[Dict] = []
    id_to_seq: Dict[str, int] = {}
    seq_counter = 0
    
    # 连接列表（存储序号对）
    bonds: List[Tuple[int, int]] = []
    bond_set: set = set()  # 用于去重
    
    def add_node(node_id: str, su_type: int, axial_coord: List[int]) -> int:
        """添加节点，返回序号"""
        nonlocal seq_counter
        
        if node_id in id_to_seq:
            return id_to_seq[node_id]
        
        seq = seq_counter
        seq_counter += 1
        
        # 由六角坐标计算笛卡尔坐标（与原始pos2d一致的布局）
        try:
            x, y = hex_to_cartesian(axial_coord[0], axial_coord[1])
        except Exception:
            x, y = 0.0, 0.0
        
        nodes.append({
            "id": node_id,
            "seq": seq,
            "su_type": su_type,
            "axial_coord": axial_coord,
            "xy_coord": [x, y],
        })
        id_to_seq[node_id] = seq
        return seq
    
    def add_bond(seq_a: int, seq_b: int):
        """添加连接（自动去重和排序）"""
        if seq_a == seq_b:
            return
        bond_key = tuple(sorted([seq_a, seq_b]))
        if bond_key not in bond_set:
            bond_set.add(bond_key)
            bonds.append(list(bond_key))
    
    # ========================================
    # 1. 处理所有团簇中的节点
    # ========================================
    clusters = mcts_json.get('clusters', [])
    
    for cluster in clusters:
        sites = cluster.get('sites', [])
        
        # 添加团簇中的所有节点
        for site in sites:
            site_uid = site.get('site_uid', '')
            su_type = site.get('su_type', 13)
            axial_coord = site.get('axial_coord', [0, 0])
            
            if site_uid:
                add_node(site_uid, su_type, axial_coord)
        
        # 添加团簇内部连接（intra_edges使用的是局部索引）
        intra_edges = cluster.get('intra_edges', [])
        for edge in intra_edges:
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                local_i, local_j = edge[0], edge[1]
                if local_i < len(sites) and local_j < len(sites):
                    uid_i = sites[local_i].get('site_uid', '')
                    uid_j = sites[local_j].get('site_uid', '')
                    if uid_i and uid_j and uid_i in id_to_seq and uid_j in id_to_seq:
                        add_bond(id_to_seq[uid_i], id_to_seq[uid_j])
    
    # ========================================
    # 2. 处理边（edges）- 包括刚性连接和柔性链
    # ========================================
    edges = mcts_json.get('edges', [])
    
    for edge in edges:
        u_site = edge.get('u_site', '')
        v_site = edge.get('v_site', '')
        chain = edge.get('chain', [])
        edge_type = edge.get('edge_type', 'rigid')
        
        # 获取端点序号（如果不存在则跳过）
        if u_site not in id_to_seq or (v_site and v_site not in id_to_seq):
            # 端点可能不存在，跳过
            if u_site and u_site not in id_to_seq:
                print(f"[Warning] u_site {u_site} not found in nodes")
            if v_site and v_site not in id_to_seq:
                print(f"[Warning] v_site {v_site} not found in nodes")
            continue
        
        u_seq = id_to_seq.get(u_site)
        v_seq = id_to_seq.get(v_site) if v_site else None
        
        if not chain:
            # 无链节点，直接连接u和v
            if u_seq is not None and v_seq is not None:
                add_bond(u_seq, v_seq)
        else:
            # 有链节点：u -> chain[0] -> chain[1] -> ... -> chain[-1] -> v
            prev_seq = u_seq
            
            for node in chain:
                if not isinstance(node, dict):
                    continue
                
                node_uid = node.get('node_uid', node.get('site_uid', node.get('uid', '')))
                su_type = node.get('su_type', 23)
                axial_coord = node.get('axial_coord', [0, 0])
                
                if not node_uid:
                    continue
                
                # 添加链节点
                curr_seq = add_node(node_uid, su_type, axial_coord)
                
                # 连接到前一个节点
                if prev_seq is not None:
                    add_bond(prev_seq, curr_seq)
                
                prev_seq = curr_seq
            
            # 连接最后一个链节点到v
            if prev_seq is not None and v_seq is not None:
                add_bond(prev_seq, v_seq)
    
    # ========================================
    # 3. 构建输出
    # ========================================
    result = {
        "nodes": nodes,
        "bonds": bonds,
        "metadata": {
            "source_hash": mcts_json.get('state_hash', ''),
            "source_csv": mcts_json.get('source_csv', ''),
            "node_count": len(nodes),
            "bond_count": len(bonds),
            "created_at": datetime.now().isoformat()
        }
    }
    
    return result


def simplify_cg_file(input_path: str, output_path: Optional[str] = None) -> Dict:
    """
    读取MCTS输出JSON文件并简化
    
    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径（可选，默认为input_simplified.json）
    
    Returns:
        简化后的JSON字典
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        mcts_json = json.load(f)
    
    result = simplify_cg_molecule(mcts_json)
    result['metadata']['source_file'] = input_path
    
    # 确定输出路径
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_simplified{ext}"
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"[CG-Simplify] Input: {input_path}")
    print(f"[CG-Simplify] Output: {output_path}")
    print(f"[CG-Simplify] Nodes: {result['metadata']['node_count']}")
    print(f"[CG-Simplify] Bonds: {result['metadata']['bond_count']}")
    
    return result


def export_to_txt(simplified: Dict, output_path: str) -> str:
    """
    将简化结果导出为TXT格式
    
    格式：
    第一部分：节点信息（按seq排序）
    第二部分：连接信息（按节点序号排序）
    
    Args:
        simplified: 简化后的JSON字典
        output_path: 输出TXT文件路径
    
    Returns:
        输出文件路径
    """
    nodes = simplified.get('nodes', [])
    bonds = simplified.get('bonds', [])
    metadata = simplified.get('metadata', {})
    
    # 按seq排序节点
    sorted_nodes = sorted(nodes, key=lambda x: x.get('seq', 0))
    
    # 按第一个节点序号排序连接，然后按第二个节点序号
    sorted_bonds = sorted(bonds, key=lambda x: (x[0], x[1]))
    
    lines = []
    
    # ========================================
    # 文件头
    # ========================================
    lines.append("# Coarse-Grained Molecule (Simplified)")
    lines.append(f"# Source: {metadata.get('source_file', 'unknown')}")
    lines.append(f"# Hash: {metadata.get('source_hash', 'unknown')}")
    lines.append(f"# Created: {metadata.get('created_at', 'unknown')}")
    lines.append(f"# Nodes: {len(nodes)}, Bonds: {len(bonds)}")
    lines.append("")
    
    # ========================================
    # 第一部分：节点信息
    # ========================================
    lines.append("=" * 70)
    lines.append("NODES")
    lines.append("=" * 70)
    lines.append(f"# {'SEQ':>4}  {'SU':>3}  {'Q':>4}  {'R':>4}  {'X':>8}  {'Y':>8}  ID")
    lines.append("-" * 70)
    
    for node in sorted_nodes:
        seq = node.get('seq', 0)
        su_type = node.get('su_type', 0)
        coord = node.get('axial_coord', [0, 0])
        q, r = coord[0], coord[1]
        xy = node.get('xy_coord', None)
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            x, y = float(xy[0]), float(xy[1])
        else:
            # 兼容旧数据：如果没有xy_coord，则现算一遍
            try:
                x, y = hex_to_cartesian(q, r)
            except Exception:
                x, y = 0.0, 0.0
        node_id = node.get('id', '')
        
        lines.append(f"  {seq:4d}  {su_type:3d}  {q:4d}  {r:4d}  {x:8.3f}  {y:8.3f}  {node_id}")
    
    lines.append("")
    
    # ========================================
    # 第二部分：连接信息
    # ========================================
    lines.append("=" * 70)
    lines.append("BONDS")
    lines.append("=" * 70)
    lines.append(f"# {'A':>4}  {'B':>4}  {'SU_A':>5}  {'SU_B':>5}  ID_A -- ID_B")
    lines.append("-" * 70)
    
    # 创建seq到node的映射
    seq_to_node = {n['seq']: n for n in nodes}
    
    for bond in sorted_bonds:
        a, b = bond[0], bond[1]
        node_a = seq_to_node.get(a, {})
        node_b = seq_to_node.get(b, {})
        
        su_a = node_a.get('su_type', 0)
        su_b = node_b.get('su_type', 0)
        id_a = node_a.get('id', '?')
        id_b = node_b.get('id', '?')
        
        lines.append(f"  {a:4d}  {b:4d}  {su_a:5d}  {su_b:5d}  {id_a} -- {id_b}")
    
    lines.append("")
    lines.append("# END")
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return output_path


def print_simplified_summary(simplified: Dict):
    """打印简化结果的摘要"""
    nodes = simplified.get('nodes', [])
    bonds = simplified.get('bonds', [])
    
    print("\n" + "=" * 60)
    print("粗粒度分子简化结果")
    print("=" * 60)
    
    # 统计SU类型分布
    su_counts: Dict[int, int] = {}
    for node in nodes:
        su = node.get('su_type', -1)
        su_counts[su] = su_counts.get(su, 0) + 1
    
    print(f"\n节点总数: {len(nodes)}")
    print(f"连接总数: {len(bonds)}")
    
    print("\nSU类型分布:")
    for su, count in sorted(su_counts.items()):
        print(f"  SU-{su:2d}: {count:3d} 个")
    
    print("\n前10个节点:")
    for node in nodes[:10]:
        print(f"  seq={node['seq']:3d}, id={node['id']:15s}, "
              f"su={node['su_type']:2d}, coord={node['axial_coord']}")
    
    print("\n前10个连接:")
    for bond in bonds[:10]:
        node_a = nodes[bond[0]] if bond[0] < len(nodes) else None
        node_b = nodes[bond[1]] if bond[1] < len(nodes) else None
        if node_a and node_b:
            print(f"  [{bond[0]:3d}]-[{bond[1]:3d}]  "
                  f"{node_a['id']} (SU-{node_a['su_type']}) -- "
                  f"{node_b['id']} (SU-{node_b['su_type']})")


# ============================================================
# 可视化功能
# ============================================================

# SU类型颜色映射
SU_COLORS = {
    # 芳香碳 (5-13) - 蓝色系
    5: '#1E90FF',   # DodgerBlue
    6: '#4169E1',   # RoyalBlue
    7: '#0000CD',   # MediumBlue
    8: '#00008B',   # DarkBlue
    9: '#6495ED',   # CornflowerBlue
    10: '#4682B4',  # SteelBlue
    11: '#5F9EA0',  # CadetBlue
    12: '#00CED1',  # DarkTurquoise
    13: '#87CEEB',  # SkyBlue
    # 脂肪链碳 (19-25) - 绿色系
    19: '#32CD32',  # LimeGreen
    20: '#228B22',  # ForestGreen
    21: '#2E8B57',  # SeaGreen
    22: '#90EE90',  # LightGreen (末端甲基)
    23: '#3CB371',  # MediumSeaGreen
    24: '#00FA9A',  # MediumSpringGreen
    25: '#006400',  # DarkGreen
    # 不饱和碳 (14-18) - 橙色系
    14: '#FF8C00',  # DarkOrange
    15: '#FFA500',  # Orange
    16: '#FFD700',  # Gold
    17: '#FF6347',  # Tomato
    18: '#FF4500',  # OrangeRed
    # 氧 (28-29) - 红色系
    28: '#DC143C',  # Crimson (羟基)
    29: '#FF0000',  # Red (醚氧)
    # 氮 (26-27) - 紫色系
    26: '#9400D3',  # DarkViolet
    27: '#8B008B',  # DarkMagenta
    # 硫 (30-31) - 黄色系
    30: '#FFD700',  # Gold
    31: '#DAA520',  # GoldenRod
    # 卤素 (32) - 灰色
    32: '#808080',  # Gray
    # 羰基/酰胺/酯 (0-4) - 粉色系
    0: '#FF69B4',   # HotPink (酰胺)
    1: '#FF1493',   # DeepPink (羧酸)
    2: '#DB7093',   # PaleVioletRed (酯)
    3: '#C71585',   # MediumVioletRed (醛酮)
    4: '#FF00FF',   # Magenta (氰基)
}

DEFAULT_COLOR = '#A9A9A9'  # DarkGray


def hex_to_cartesian(q: int, r: int) -> Tuple[float, float]:
    """六角坐标转笛卡尔坐标（与原始pos2d一致的投影）"""
    import math
    # 原始构建器使用的二维投影等价于：
    #   x = (sqrt(3) / 2) * q
    #   y = r - 0.5 * q
    # 这样 r 轴近似竖直，q 轴与其成 120°，与原始六角坐标图的布局一致
    x = (math.sqrt(3) / 2.0) * q
    y = r - 0.5 * q
    return x, y


def visualize_cg_molecule(simplified: Dict, output_path: str, 
                          figsize: Tuple[int, int] = (20, 16),
                          node_size: int = 300,
                          font_size: int = 8,
                          show_id: bool = False) -> str:
    """
    可视化粗粒度分子图
    
    Args:
        simplified: 简化后的JSON字典
        output_path: 输出图片路径
        figsize: 图片尺寸
        node_size: 节点大小
        font_size: 字体大小
        show_id: 是否显示节点ID（默认只显示SU类型）
    
    Returns:
        输出文件路径
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[Error] matplotlib is required for visualization. Install with: pip install matplotlib")
        return ""
    
    nodes = simplified.get('nodes', [])
    bonds = simplified.get('bonds', [])
    
    if not nodes:
        print("[Warning] No nodes to visualize")
        return ""
    
    # 计算笛卡尔坐标
    coords = {}
    for node in nodes:
        seq = node['seq']
        q, r = node['axial_coord']
        x, y = hex_to_cartesian(q, r)
        coords[seq] = (x, y)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制连接线
    for bond in bonds:
        a, b = bond[0], bond[1]
        if a in coords and b in coords:
            x1, y1 = coords[a]
            x2, y2 = coords[b]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.8, alpha=0.6, zorder=1)
    
    # 绘制节点
    for node in nodes:
        seq = node['seq']
        su_type = node['su_type']
        x, y = coords[seq]
        
        # 获取颜色
        color = SU_COLORS.get(su_type, DEFAULT_COLOR)
        
        # 绘制圆圈
        circle = plt.Circle((x, y), 0.35, color=color, ec='black', linewidth=0.5, zorder=2)
        ax.add_patch(circle)
        
        # 标注SU类型
        if show_id:
            label = f"{su_type}\n{seq}"
        else:
            label = str(su_type)
        ax.text(x, y, label, ha='center', va='center', fontsize=font_size, 
                fontweight='bold', color='white', zorder=3)
    
    # 设置坐标轴
    ax.set_aspect('equal')
    ax.autoscale()
    
    # 添加边距
    x_coords = [c[0] for c in coords.values()]
    y_coords = [c[1] for c in coords.values()]
    margin = 2
    ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
    ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
    
    # 隐藏坐标轴
    ax.axis('off')
    
    # 添加图例
    legend_handles = []
    su_types_in_mol = sorted(set(n['su_type'] for n in nodes))
    
    for su in su_types_in_mol:
        color = SU_COLORS.get(su, DEFAULT_COLOR)
        patch = mpatches.Patch(color=color, label=f'SU-{su}')
        legend_handles.append(patch)
    
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), 
              fontsize=8, title='SU Types', title_fontsize=10)
    
    # 添加标题
    metadata = simplified.get('metadata', {})
    title = f"Coarse-Grained Molecule\nNodes: {len(nodes)}, Bonds: {len(bonds)}"
    if metadata.get('source_hash'):
        title += f"\nHash: {metadata['source_hash']}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path


def visualize_from_txt(txt_path: str, output_path: Optional[str] = None) -> str:
    """
    从TXT文件读取并可视化粗粒度分子
    
    Args:
        txt_path: TXT文件路径
        output_path: 输出图片路径（可选）
    
    Returns:
        输出文件路径
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"TXT file not found: {txt_path}")
    
    nodes = []
    bonds = []
    metadata = {}
    
    current_section = None
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 解析元数据
            if line.startswith('# Source:'):
                metadata['source_file'] = line.split(':', 1)[1].strip()
            elif line.startswith('# Hash:'):
                metadata['source_hash'] = line.split(':', 1)[1].strip()
            elif line.startswith('# Nodes:'):
                pass  # 忽略统计行
            
            # 识别段落
            if line == 'NODES':
                current_section = 'nodes'
                continue
            elif line == 'BONDS':
                current_section = 'bonds'
                continue
            elif line.startswith('=') or line.startswith('-') or line.startswith('#'):
                continue
            elif line == '# END':
                break
            
            # 解析数据
            if current_section == 'nodes' and line:
                parts = line.split()
                # 兼容两种格式：
                # 旧格式: SEQ SU Q R ID
                # 新格式: SEQ SU Q R X Y ID
                if len(parts) >= 5:
                    try:
                        seq = int(parts[0])
                        su_type = int(parts[1])
                        q = int(parts[2])
                        r = int(parts[3])
                        x = y = None
                        node_id = ""

                        if len(parts) >= 7:
                            # 新格式，解析X,Y，并将最后一列作为ID
                            x = float(parts[4])
                            y = float(parts[5])
                            node_id = parts[6]
                        else:
                            # 旧格式，没有X,Y
                            node_id = parts[4]

                        node = {
                            'seq': seq,
                            'su_type': su_type,
                            'axial_coord': [q, r],
                            'id': node_id,
                        }
                        if x is not None and y is not None:
                            node['xy_coord'] = [x, y]

                        nodes.append(node)
                    except ValueError:
                        continue
            
            elif current_section == 'bonds' and line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        a = int(parts[0])
                        b = int(parts[1])
                        bonds.append([a, b])
                    except ValueError:
                        continue
    
    # 构建简化数据结构
    simplified = {
        'nodes': nodes,
        'bonds': bonds,
        'metadata': metadata
    }
    
    # 确定输出路径
    if output_path is None:
        base = os.path.splitext(txt_path)[0]
        output_path = f"{base}.png"
    
    return visualize_cg_molecule(simplified, output_path)


# ============================================================
# 命令行入口
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='简化MCTS输出的粗粒度分子JSON，或从TXT可视化')
    parser.add_argument('input', help='输入文件路径：原始JSON（clusters+edges）或简化TXT')
    parser.add_argument('-o', '--output', help='输出JSON文件路径（仅JSON输入时有效，可选）')
    parser.add_argument('-t', '--txt', action='store_true', help='JSON输入时同时输出TXT格式')
    parser.add_argument('-p', '--plot', action='store_true', help='生成可视化图片')
    parser.add_argument('-v', '--verbose', action='store_true', help='打印详细信息')
    
    args = parser.parse_args()
    ext = os.path.splitext(args.input)[1].lower()

    # ===========================================================
    # 情况 1：输入为简化的TXT文件，仅做可视化检查
    # ===========================================================
    if ext == '.txt':
        txt_path = args.input
        if args.plot:
            png_path = os.path.splitext(txt_path)[0] + '.png'
            visualize_from_txt(txt_path, png_path)
            print(f"[CG-Simplify] Plot from TXT: {png_path}")

        if args.txt:
            # 已经是TXT，不再重复导出
            print("[CG-Simplify] 输入已是TXT，跳过TXT导出")

        if args.verbose:
            # 从TXT还原简化数据，打印摘要
            simplified_from_txt = {
                'nodes': [],
                'bonds': [],
                'metadata': {},
            }
            try:
                # 复用visualize_from_txt里的解析逻辑较复杂，这里简单提示
                print("[CG-Simplify] TXT输入：已生成可视化（如指定 -p）")
            except Exception:
                pass
    
    # ===========================================================
    # 情况 2：输入为原始JSON，先简化再导出/可视化
    # ===========================================================
    else:
        result = simplify_cg_file(args.input, args.output)
        
        # 输出TXT格式
        if args.txt:
            base = os.path.splitext(args.output or args.input)[0]
            if base.endswith('_simplified'):
                txt_path = f"{base}.txt"
            else:
                txt_path = f"{base}_simplified.txt"
            export_to_txt(result, txt_path)
            print(f"[CG-Simplify] TXT: {txt_path}")
        
        # 生成可视化图片（基于简化后的nodes+bonds）
        if args.plot:
            base = os.path.splitext(args.output or args.input)[0]
            if base.endswith('_simplified'):
                png_path = f"{base}.png"
            else:
                png_path = f"{base}_simplified.png"
            visualize_cg_molecule(result, png_path)
            print(f"[CG-Simplify] Plot: {png_path}")
        
        if args.verbose:
            print_simplified_summary(result)
