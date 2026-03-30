#!/usr/bin/env python3
"""
serialization.py - 粗粒度分子序列化模块

将Builder对象序列化为JSON格式，用于MCTS搜索结果的保存和加载。
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional


def serialize_builder(builder) -> Dict:
    """
    将Builder对象序列化为JSON格式的字典
    
    Args:
        builder: AromaticFlexibleBuilder或其子类实例
    
    Returns:
        可JSON序列化的字典
    """
    result = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "clusters": [],
        "edges": [],
    }
    
    # ========================================
    # 1. 序列化团簇
    # ========================================
    if hasattr(builder, 'clusters'):
        for cluster in builder.clusters:
            cluster_data = serialize_cluster(cluster)
            if cluster_data:
                result["clusters"].append(cluster_data)
    
    # ========================================
    # 2. 序列化边（刚性连接、柔性链、侧链）
    # ========================================
    
    # 刚性连接
    if hasattr(builder, 'rigid_edges'):
        for edge in builder.rigid_edges:
            edge_data = serialize_rigid_edge(edge)
            if edge_data:
                result["edges"].append(edge_data)
    
    # 柔性链
    if hasattr(builder, 'flex_edges'):
        for edge in builder.flex_edges:
            edge_data = serialize_flex_edge(edge, "flex")
            if edge_data:
                result["edges"].append(edge_data)
    
    # 侧链
    if hasattr(builder, 'side_chains'):
        for edge in builder.side_chains:
            edge_data = serialize_flex_edge(edge, "side")
            if edge_data:
                result["edges"].append(edge_data)
    
    return result


def serialize_cluster(cluster) -> Optional[Dict]:
    """序列化单个团簇"""
    if cluster is None:
        return None
    
    try:
        cluster_data = {
            "cluster_id": getattr(cluster, 'cluster_id', 0),
            "kind": getattr(cluster, 'kind', 'benzene'),
            "ring_count": getattr(cluster, 'ring_count', 1),
            "sites": [],
            "ring_centers": [],
            "intra_edges": [],
            "translation": getattr(cluster, 'translation', [0, 0]),
        }
        
        # 序列化sites
        if hasattr(cluster, 'sites'):
            for i, site in enumerate(cluster.sites):
                site_data = serialize_site(site, cluster.cluster_id, i)
                if site_data:
                    cluster_data["sites"].append(site_data)
        
        # 序列化ring_centers
        if hasattr(cluster, 'ring_centers'):
            for center in cluster.ring_centers:
                if hasattr(center, '__iter__'):
                    cluster_data["ring_centers"].append(list(center))
                else:
                    cluster_data["ring_centers"].append([center, 0])
        
        # 序列化intra_edges（内部连接）
        if hasattr(cluster, 'intra_edges'):
            for edge in cluster.intra_edges:
                if hasattr(edge, '__iter__') and len(edge) >= 2:
                    cluster_data["intra_edges"].append([edge[0], edge[1]])
        elif hasattr(cluster, 'edges'):
            for edge in cluster.edges:
                if hasattr(edge, '__iter__') and len(edge) >= 2:
                    cluster_data["intra_edges"].append([edge[0], edge[1]])
        
        return cluster_data
        
    except Exception as e:
        print(f"[Warning] Failed to serialize cluster: {e}")
        return None


def serialize_site(site, cluster_id: int, local_idx: int) -> Optional[Dict]:
    """序列化单个site"""
    if site is None:
        return None
    
    try:
        # 获取site_uid
        if hasattr(site, 'site_uid'):
            site_uid = site.site_uid
        elif hasattr(site, 'uid'):
            site_uid = site.uid
        else:
            site_uid = f"C{cluster_id}-V{local_idx}"
        
        # 获取axial_coord
        if hasattr(site, 'axial_coord'):
            axial_coord = list(site.axial_coord) if hasattr(site.axial_coord, '__iter__') else [site.axial_coord, 0]
        elif hasattr(site, 'coord'):
            axial_coord = list(site.coord) if hasattr(site.coord, '__iter__') else [site.coord, 0]
        else:
            axial_coord = [0, 0]
        
        # 获取pos2d
        if hasattr(site, 'pos2d'):
            pos2d = list(site.pos2d) if hasattr(site.pos2d, '__iter__') else [site.pos2d, 0]
        else:
            pos2d = [0.0, 0.0]
        
        site_data = {
            "site_uid": site_uid,
            "cluster_id": cluster_id,
            "su_type": getattr(site, 'su_type', 13),
            "orbit_id": getattr(site, 'orbit_id', 0),
            "occupied": getattr(site, 'occupied', False),
            "axial_coord": axial_coord,
            "pos2d": pos2d,
            "ring_refs": list(getattr(site, 'ring_refs', [])) if hasattr(site, 'ring_refs') else [],
        }
        
        return site_data
        
    except Exception as e:
        print(f"[Warning] Failed to serialize site: {e}")
        return None


def serialize_rigid_edge(edge) -> Optional[Dict]:
    """序列化刚性边"""
    if edge is None:
        return None
    
    try:
        edge_data = {
            "edge_type": "rigid",
            "u_site": getattr(edge, 'u_site', '') or getattr(edge, 'u', ''),
            "v_site": getattr(edge, 'v_site', '') or getattr(edge, 'v', ''),
            "chain": [],
        }
        return edge_data
        
    except Exception as e:
        print(f"[Warning] Failed to serialize rigid edge: {e}")
        return None


def serialize_flex_edge(edge, edge_type: str = "flex") -> Optional[Dict]:
    """序列化柔性边（柔性链或侧链）"""
    if edge is None:
        return None
    
    try:
        # 获取端点
        u_site = getattr(edge, 'u_site', '') or getattr(edge, 'u', '') or getattr(edge, 'start_site', '')
        v_site = getattr(edge, 'v_site', None) or getattr(edge, 'v', None) or getattr(edge, 'end_site', None)
        
        edge_data = {
            "edge_type": edge_type,
            "u_site": u_site,
            "v_site": v_site,
            "chain": [],
        }
        
        # 序列化链节点
        chain = getattr(edge, 'chain', []) or getattr(edge, 'nodes', [])
        for i, node in enumerate(chain):
            node_data = serialize_chain_node(node, u_site, i)
            if node_data:
                edge_data["chain"].append(node_data)
        
        return edge_data
        
    except Exception as e:
        print(f"[Warning] Failed to serialize flex edge: {e}")
        return None


def serialize_chain_node(node, parent_uid: str, idx: int) -> Optional[Dict]:
    """序列化链节点"""
    if node is None:
        return None
    
    try:
        # 获取node_uid
        if hasattr(node, 'node_uid'):
            node_uid = node.node_uid
        elif hasattr(node, 'uid'):
            node_uid = node.uid
        else:
            node_uid = f"{parent_uid}-{idx}"
        
        # 获取axial_coord
        if hasattr(node, 'axial_coord'):
            axial_coord = list(node.axial_coord) if hasattr(node.axial_coord, '__iter__') else [node.axial_coord, 0]
        elif hasattr(node, 'coord'):
            axial_coord = list(node.coord) if hasattr(node.coord, '__iter__') else [node.coord, 0]
        else:
            axial_coord = [0, 0]
        
        # 获取pos2d
        if hasattr(node, 'pos2d'):
            pos2d = list(node.pos2d) if hasattr(node.pos2d, '__iter__') else [node.pos2d, 0]
        else:
            pos2d = [0.0, 0.0]
        
        node_data = {
            "node_uid": node_uid,
            "su_type": getattr(node, 'su_type', 23),
            "axial_coord": axial_coord,
            "pos2d": pos2d,
        }
        
        return node_data
        
    except Exception as e:
        print(f"[Warning] Failed to serialize chain node: {e}")
        return None


def save_coarse_grained_molecule(
    builder,
    output_path: str,
    source_csv: str = "",
    state_hash: str = "",
    score: float = 0.0,
    info: Optional[Dict] = None,
) -> bool:
    """
    将粗粒度分子保存到JSON文件
    
    Args:
        builder: Builder对象
        output_path: 输出文件路径
        source_csv: 源CSV文件路径
        state_hash: 状态哈希
        score: 分数
        info: 额外信息
    
    Returns:
        是否成功
    """
    try:
        result = serialize_builder(builder)
        result["source_csv"] = source_csv
        result["state_hash"] = state_hash
        result["score"] = score
        
        if info:
            result["info"] = info
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"[Error] Failed to save coarse-grained molecule: {e}")
        return False


def load_coarse_grained_molecule(input_path: str) -> Optional[Dict]:
    """
    从JSON文件加载粗粒度分子数据
    
    Args:
        input_path: 输入文件路径
    
    Returns:
        粗粒度分子数据字典
    """
    if not os.path.exists(input_path):
        print(f"[Error] File not found: {input_path}")
        return None
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load file: {e}")
        return None


def get_molecule_summary(data: Dict) -> Dict:
    """获取分子摘要信息"""
    clusters = data.get('clusters', [])
    edges = data.get('edges', [])
    
    # 统计节点
    node_count = sum(len(c.get('sites', [])) for c in clusters)
    for edge in edges:
        node_count += len(edge.get('chain', []))
    
    # 统计SU类型
    su_counts = {}
    for cluster in clusters:
        for site in cluster.get('sites', []):
            su = site.get('su_type', -1)
            su_counts[su] = su_counts.get(su, 0) + 1
    for edge in edges:
        for node in edge.get('chain', []):
            su = node.get('su_type', -1)
            su_counts[su] = su_counts.get(su, 0) + 1
    
    # 统计边类型
    edge_types = {}
    for edge in edges:
        et = edge.get('edge_type', 'unknown')
        edge_types[et] = edge_types.get(et, 0) + 1
    
    return {
        'cluster_count': len(clusters),
        'node_count': node_count,
        'edge_count': len(edges),
        'su_distribution': su_counts,
        'edge_type_distribution': edge_types,
        'score': data.get('score', 0),
    }


def print_molecule_summary(data: Dict):
    """打印分子摘要"""
    summary = get_molecule_summary(data)
    
    print("\n" + "=" * 60)
    print("粗粒度分子摘要")
    print("=" * 60)
    print(f"团簇数: {summary['cluster_count']}")
    print(f"节点数: {summary['node_count']}")
    print(f"边数: {summary['edge_count']}")
    print(f"分数: {summary['score']:.4f}")
    
    print("\nSU类型分布:")
    for su, count in sorted(summary['su_distribution'].items()):
        print(f"  SU-{su:2d}: {count:3d}")
    
    print("\n边类型分布:")
    for et, count in sorted(summary['edge_type_distribution'].items()):
        print(f"  {et}: {count}")
