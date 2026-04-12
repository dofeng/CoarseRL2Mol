"""
Inverse Pipeline Layer0 Module - Layer0: SU直方图估计器
"""
import math
import torch
from typing import Tuple, Dict

from .inverse_common import get_aliphatic_carbon_policy

class Layer0Estimator:
    """Layer0直方图估计器"""
    def __init__(self, s2n_model, E_SU_tensor: torch.Tensor, device: str = 'cpu'):
        self.s2n = s2n_model
        self.E_SU = E_SU_tensor.to(device)
        self.device = device

    @staticmethod
    def _nearest_even_int(value: float) -> int:
        value = max(0.0, float(value))
        lower = int(math.floor(value))
        if lower % 2 != 0:
            lower -= 1
        upper = int(math.ceil(value))
        if upper % 2 != 0:
            upper += 1
        lower = max(0, lower)
        upper = max(0, upper)
        if abs(float(value) - float(lower)) <= abs(float(upper) - float(value)):
            return int(lower)
        return int(upper)

    @staticmethod
    def _allocate_ratio_counts(total: int, ratios: Tuple[float, ...]) -> Tuple[int, ...]:
        total = max(0, int(total))
        if total <= 0:
            return tuple(0 for _ in ratios)
        raw = [max(0.0, float(r)) * float(total) for r in ratios]
        base = [int(math.floor(v)) for v in raw]
        remainder = int(total - sum(base))
        if remainder > 0:
            order = sorted(
                range(len(raw)),
                key=lambda i: (raw[i] - float(base[i]), raw[i]),
                reverse=True,
            )
            for i in range(remainder):
                base[order[i % len(order)]] += 1
        return tuple(int(v) for v in base)

    def _reconcile_carbon_total(self, H: torch.Tensor, S_target: torch.Tensor, E_target: torch.Tensor) -> torch.Tensor:
        """最终修正C总量，只在纯碳结构单元中调节，避免破坏 O/N/S/X 守恒。"""
        H_new = H.clone()
        try:
            E_curr = torch.matmul(H_new.float(), self.E_SU.to(H_new.device))
            target_C = int(E_target[0].item())
            current_C = int(round(float(E_curr[0].item())))
        except Exception:
            return H_new

        delta_C = int(target_C - current_C)
        if delta_C == 0:
            return H_new

        spectrum = S_target.detach().cpu().numpy()
        total_area = float(spectrum.sum() * 0.1)
        aliphatic_area = float(spectrum[:900].sum() * 0.1)
        aromatic_area = float(spectrum[900:1600].sum() * 0.1)
        if total_area > 1e-6:
            x = float(aliphatic_area / total_area)
            y = float(aromatic_area / total_area)
        else:
            x, y = 0.33, 0.33

        if delta_C > 0:
            add_aro = int(round(float(delta_C) * y / max(1e-6, x + y)))
            add_ali = int(round(float(delta_C) * x / max(1e-6, x + y)))
            add_uns = int(delta_C - add_aro - add_ali)
            H_new[13] += max(0, add_aro)
            H_new[23] += max(0, add_ali)
            H_new[15] += max(0, add_uns)
        else:
            deficit = int(-delta_C)
            removal_order = [13, 23, 15, 12, 10, 24, 22, 14, 16, 17, 18, 25, 11]
            for su in removal_order:
                if deficit <= 0:
                    break
                available = int(H_new[su].item())
                if available <= 0:
                    continue
                take = min(available, deficit)
                H_new[su] -= int(take)
                deficit -= int(take)

        return torch.clamp(H_new, min=0).long()

    @staticmethod
    def _compute_region_area_ratios(S_target: torch.Tensor) -> Tuple[float, float, float]:
        spectrum = S_target.detach().cpu().numpy()
        total_area = float(spectrum.sum() * 0.1)

        aliphatic_area = float(spectrum[:900].sum() * 0.1)
        aromatic_area = float(spectrum[900:1600].sum() * 0.1)
        carbonyl_area = float(spectrum[1600:].sum() * 0.1)

        if total_area > 1e-6:
            x = float(aliphatic_area / total_area)
            y = float(aromatic_area / total_area)
            z = float(carbonyl_area / total_area)
        else:
            x, y, z = 0.33, 0.33, 0.34
        return x, y, z

    def _estimate_region_carbon_budgets(self,
                                        S_target: torch.Tensor,
                                        E_target: torch.Tensor) -> Dict[str, float]:
        """
        根据谱图三个区域面积比例估计 C 元素预算。

        约定:
          x: 0-90 ppm 脂肪区面积占比
          y: 90-160 ppm 芳香/非饱和区面积占比
          z: 160-240 ppm 羰基区面积占比

        碳预算:
          aliphatic_C = scale(H/C) * x * N
          carbonyl_C = 1.35 * z * N
          aromatic_C = N - aliphatic_C - carbonyl_C
        """
        x, y, z = self._compute_region_area_ratios(S_target)
        total_C = float(E_target[0].item())
        xN = float(x) * float(total_C)
        yN = float(y) * float(total_C)
        zN = float(z) * float(total_C)

        policy = get_aliphatic_carbon_policy(E_target)
        aliphatic_scale = float(policy.get('init_aliphatic_scale', 0.82))
        aliphatic_C = max(0.0, float(aliphatic_scale) * float(xN))
        carbonyl_C = max(0.0, 1.35 * float(zN))
        aromatic_C = max(0.0, float(total_C) - float(aliphatic_C) - float(carbonyl_C))

        return {
            'x': float(x),
            'y': float(y),
            'z': float(z),
            'N': float(total_C),
            'xN': float(xN),
            'yN': float(yN),
            'zN': float(zN),
            'aliphatic_C': float(aliphatic_C),
            'aromatic_C': float(aromatic_C),
            'carbonyl_C': float(carbonyl_C),
            'hc_ratio': float(policy.get('hc_ratio', 0.0)),
            'init_aliphatic_scale': float(aliphatic_scale),
            'layer4_aliphatic_upper_scale': float(policy.get('layer4_aliphatic_upper_scale', 0.90)),
        }

    def estimate_su_histogram(self, S_target: torch.Tensor, 
                               E_target: torch.Tensor) -> torch.Tensor:
        """
        Layer0: 从谱图和元素推断初始SU直方图
        
        改进点：
        1. 修正顺序优化：X → S → N → C=O羰基 → O → 连接匹配
        2. 基于光谱区域比例修正羰基分布
        3. 多样性惩罚：防止某些SU过度集中
        """
        device = self.device
        S_target = S_target.to(device)
        E_target = E_target.to(device)
        
        print("\n[Layer0] SU直方图估计开始")
        
        # Step 1: S2N模型预测初始SU分布
        with torch.no_grad():
            if hasattr(self.s2n, 'infer_su_hist'):
                H_pred = self.s2n.infer_su_hist(S_target.unsqueeze(0), E_target.unsqueeze(0)).squeeze(0)
            else:
                H_pred = torch.nn.functional.softplus(
                    self.s2n(S_target.unsqueeze(0), E_target.unsqueeze(0))
                ).squeeze(0)
        
        # 整数化（四舍五入）
        H_init = torch.round(H_pred).long()
        H_init = torch.clamp(H_init, min=0)
        
        # Step 2: 杂原子元素修正（X → S → N 顺序）
        H_corrected = H_init.clone()
        
        # 2.1 修正X元素（32号）
        H_corrected = self._correct_halogen_X(H_corrected, E_target)
        
        # 2.2 修正S元素（30, 31号）
        H_corrected = self._correct_sulfur_S(H_corrected, E_target)
        
        # 2.3 修正N元素（0, 4, 26, 27号）
        H_corrected = self._correct_nitrogen_N(H_corrected, E_target)
        
        # Step 3: C=O羰基分布修正（基于光谱区域比例）
        self._carbonyl_shortage = 0
        self._o_cap_triggered = False
        H_corrected = self._correct_carbonyl_distribution(H_corrected, S_target, E_target)
        
        # Step 4: O元素修正（只修正28, 29号）
        target_O = int(E_target[2].item())
        used_O_03 = (
            int(H_corrected[0].item())
            + 2 * int(H_corrected[1].item())
            + 2 * int(H_corrected[2].item())
            + int(H_corrected[3].item())
        )
        if bool(getattr(self, '_o_cap_triggered', False)) or used_O_03 >= target_O:
            H_corrected = H_corrected.clone()
            H_corrected[28] = 0
            H_corrected[29] = 0
        else:
            H_corrected = self._correct_oxygen_O(H_corrected, E_target)
        
        # Step 5: 含碳结构单元连接匹配修正
        
        # 3.1 修正C=O连接（9号）
        H_corrected = self._correct_carbonyl_connection(H_corrected)
        
        # 3.2 修正-O-连接（5号、19号）
        H_corrected, ether_meta = self._correct_ether_connection(H_corrected)
        
        # 3.3 修正-S-连接（7号），传入O基准
        o_base_19 = ether_meta.get('o_base_19', 0)
        H_corrected, _ = self._correct_thioether_connection(H_corrected, o_base_19)
        
        # 3.4 修正-NH-连接（6号、20号）
        H_corrected = self._correct_amine_connection(H_corrected)
        
        # 3.5 修正-X连接（8号、21号）
        H_corrected = self._correct_halogen_connection(H_corrected)
        
        # Step 6: 脂肪碳结构修正（22, 23, 24, 25号）
        H_corrected = self._correct_aliphatic_carbons(H_corrected, S_target, E_target)
        
        # Step 7: 非饱和结构修正（14, 15, 16, 17, 18号）
        H_corrected = self._correct_unsaturated_carbons(H_corrected, S_target, E_target)
        
        # Step 8: 芳香结构修正（10, 11, 12, 13号）
        H_tmp = self._correct_aromatic_carbons(H_corrected, S_target, E_target)
        if H_tmp is not None:
            H_corrected = H_tmp

        # Step 8.5: 最终碳元素守恒修正
        H_tmp = self._reconcile_carbon_total(H_corrected, S_target, E_target)
        if H_tmp is not None:
            H_corrected = H_tmp
            
        # Step 9: H元素调整（三区域调整）
        H_tmp = self._adjust_hydrogen(H_corrected, E_target)
        if H_tmp is not None:
            H_corrected = H_tmp
            
        # 确保所有值为非负整数
        H_corrected = torch.clamp(H_corrected, min=0).long()
            
        print(f"[Layer0] 完成 - 总SU={int(H_corrected.sum().item())}")
        
        return H_corrected
    
    def _correct_halogen_X(self, H: torch.Tensor, E_target: torch.Tensor) -> torch.Tensor:
        """
        修正X元素（32号）
        直接调整到目标X数量
        """
        target_X = int(E_target[5].item())
        current_X = int(H[32].item())

        if current_X == target_X:
            return H
        
        H_new = H.clone()
        
        if current_X > target_X:
            # 删除多余的32号
            H_new[32] = target_X
        else:
            # 补充缺少的32号
            H_new[32] = target_X
        
        return H_new
    
    def _correct_sulfur_S(self, H: torch.Tensor, E_target: torch.Tensor) -> torch.Tensor:
        """
        修正S元素（30, 31号）
        - 总数 = 目标S
        - 优先级：31 > 30
        - 目标比例：30:31 = 0.4:0.6
        """
        target_S = int(E_target[4].item())
        m = int(H[30].item())  # 30号当前数量
        n = int(H[31].item())  # 31号当前数量
        current_S = m + n

        if current_S == target_S:
            return H
        
        H_new = H.clone()
        
        # 目标分布：30:31 = 0.4:0.6
        target_30 = target_S * 0.4
        target_31 = target_S * 0.6
        
        if current_S < target_S:
            # 需要补充
            diff = target_S - current_S
            
            for _ in range(diff):
                # 计算当前偏差
                delta_30 = m - target_30
                delta_31 = n - target_31
                
                # 补充负值最大的；偏差相同时优先补充优先级高的（31 > 30）
                if delta_30 < delta_31:
                    m += 1
                elif delta_31 < delta_30:
                    n += 1
                else:  # 偏差相同，优先补充31号
                    n += 1
        else:
            # 需要删除
            diff = current_S - target_S
            
            for _ in range(diff):
                # 计算当前偏差
                delta_30 = m - target_30
                delta_31 = n - target_31
                
                # 删除正值最大的；偏差相同时优先删除优先级低的（30 < 31）
                if delta_30 > delta_31 and m > 0:
                    m -= 1
                elif delta_31 > delta_30 and n > 0:
                    n -= 1
                elif delta_30 == delta_31:  # 偏差相同，优先删除30号
                    if m > 0:
                        m -= 1
                    elif n > 0:
                        n -= 1
                elif m > 0:  # 兜底
                    m -= 1
                elif n > 0:
                    n -= 1
        
        H_new[30] = m
        H_new[31] = n
        return H_new
    
    def _correct_nitrogen_N(self, H: torch.Tensor, E_target: torch.Tensor) -> torch.Tensor:
        """
        修正N元素（0, 4, 26, 27号）
        """
        target_N = int(E_target[3].item())
        x = int(H[0].item())   
        y = int(H[4].item())   
        z = int(H[26].item())  
        w = int(H[27].item())  
        current_N = x + y + z + w

        if current_N == target_N:
            return H
        
        H_new = H.clone()
        
        # 目标分布：0:4:26:27 = 0.1:0.05:0.45:0.4
        target_0 = target_N * 0.1
        target_4 = target_N * 0.05
        target_26 = target_N * 0.45
        target_27 = target_N * 0.4
        
        if current_N < target_N:
            # 需要补充
            diff = target_N - current_N
            
            for _ in range(diff):
                # 计算当前偏差
                delta_0 = x - target_0
                delta_4 = y - target_4
                delta_26 = z - target_26
                delta_27 = w - target_27

                priority = {26: -4, 27: -3, 0: -2, 4: -1}
                deltas = [(delta_0, priority[0], 0, '0号'), 
                          (delta_4, priority[4], 4, '4号'), 
                          (delta_26, priority[26], 26, '26号'), 
                          (delta_27, priority[27], 27, '27号')]
                _, _, su_idx, su_name = min(deltas, key=lambda t: (t[0], t[1]))
                
                # 补充该结构单元
                if su_idx == 0:
                    x += 1
                elif su_idx == 4:
                    y += 1
                elif su_idx == 26:
                    z += 1
                else:  # 27
                    w += 1
        else:
            # 需要删除
            diff = current_N - target_N
            
            for _ in range(diff):
                # 计算当前偏差
                delta_0 = x - target_0
                delta_4 = y - target_4
                delta_26 = z - target_26
                delta_27 = w - target_27

                priority = {26: 4, 27: 3, 0: 2, 4: 1}
                candidates = []
                if x > 0:
                    candidates.append((delta_0, -priority[0], 0, '0号'))
                if y > 0:
                    candidates.append((delta_4, -priority[4], 4, '4号'))
                if z > 0:
                    candidates.append((delta_26, -priority[26], 26, '26号'))
                if w > 0:
                    candidates.append((delta_27, -priority[27], 27, '27号'))
                
                if not candidates:
                    break
                
                _, _, su_idx, su_name = max(candidates, key=lambda t: (t[0], t[1]))
                
                # 删除该结构单元
                if su_idx == 0:
                    x -= 1
                elif su_idx == 4:
                    y -= 1
                elif su_idx == 26:
                    z -= 1
                else:  # 27
                    w -= 1
                        
        H_new[0] = x
        H_new[4] = y
        H_new[26] = z
        H_new[27] = w
              
        return H_new
    
    def _correct_carbonyl_distribution(self, H: torch.Tensor, S_target: torch.Tensor,
                                       E_target: torch.Tensor) -> torch.Tensor:
        """
        修正C=O羰基分布（1, 2, 3号）
        """
        budgets = self._estimate_region_carbon_budgets(S_target, E_target)
        carbonyl_C = int(round(float(budgets['carbonyl_C'])))
        
        # 0号由N修正固定，保持不变
        n_0 = int(H[0].item())
        
        # 需要修正的羰基结构数量
        W = carbonyl_C - n_0
        W = max(0, W)  # 防止负数

        if W == 0:
            self._carbonyl_shortage = 0
            self._o_cap_triggered = False
            return H
        
        H_new = H.clone()
        
        # 当前1、2、3号数量
        m = int(H[1].item())  # 1号 (COOH)
        n = int(H[2].item())  # 2号 (-COO-)
        p = int(H[3].item())  # 3号 (-C=O-)
        
        current_total = m + n + p
        
        # 目标分布：1:2:3 = 0.35:0.25:0.4
        target_1 = W * 0.35
        target_2 = W * 0.25
        target_3 = W * 0.4
        
        if current_total == W:
            return H_new
        
        if current_total < W:
            # 需要补充
            diff = W - current_total
            
            for _ in range(diff):
                delta_1 = m - target_1
                delta_2 = n - target_2
                delta_3 = p - target_3
                
                # 补充负值最大的；偏差相同时优先补充优先级高的（3 > 1 > 2）
                priority = {3: -3, 1: -2, 2: -1}
                candidates = [
                    (delta_1, priority[1], 1, '1号'),
                    (delta_2, priority[2], 2, '2号'),
                    (delta_3, priority[3], 3, '3号')
                ]
                min_delta, _, su_idx, su_name = min(candidates, key=lambda t: (t[0], t[1]))
                
                if su_idx == 1:
                    m += 1
                elif su_idx == 2:
                    n += 1
                else:  # 3
                    p += 1
        else:
            # 需要删除
            diff = current_total - W
            
            for _ in range(diff):
                delta_1 = m - target_1
                delta_2 = n - target_2
                delta_3 = p - target_3
                
                # 删除正值最大的；偏差相同时优先删除优先级低的（2 < 1 < 3）
                priority = {3: 3, 1: 2, 2: 1}
                candidates = []
                if m > 0:
                    candidates.append((delta_1, -priority[1], 1, '1号'))
                if n > 0:
                    candidates.append((delta_2, -priority[2], 2, '2号'))
                if p > 0:
                    candidates.append((delta_3, -priority[3], 3, '3号'))
                
                if not candidates:
                    break
                
                max_delta, _, su_idx, su_name = max(candidates, key=lambda t: (t[0], t[1]))
                
                if su_idx == 1:
                    m -= 1
                elif su_idx == 2:
                    n -= 1
                else:  # 3
                    p -= 1
        
        H_new[1] = m
        H_new[2] = n
        H_new[3] = p

        target_O = int(E_target[2].item())
        used_O_03 = int(n_0 + 2 * m + 2 * n + p)
        if used_O_03 > target_O:

            self._o_cap_triggered = True
            Y = int(target_O - n_0)
            Y = max(0, Y)

            W_total = int(carbonyl_C)
            target_o1 = float(0.35 * W_total)
            target_o2 = float(0.25 * W_total)
            target_o3 = float(0.4 * W_total)

            delete_priority = {1: 0, 2: 1, 3: 2}
            max_iters = int(max(100, used_O_03 - target_O + 50))
            iters = 0
            while (2 * m + 2 * n + p) > Y and iters < max_iters:
                iters += 1
                cur_o = int(2 * m + 2 * n + p)
                need_remove = int(cur_o - Y)

                d1 = float(2 * m) - target_o1
                d2 = float(2 * n) - target_o2
                d3 = float(p) - target_o3

                candidates = []
                if m > 0 and (cur_o - 2) >= Y:
                    candidates.append((d1, delete_priority[1], 1))
                if n > 0 and (cur_o - 2) >= Y:
                    candidates.append((d2, delete_priority[2], 2))
                if p > 0 and (cur_o - 1) >= Y:
                    candidates.append((d3, delete_priority[3], 3))

                if not candidates:
                    break

                if need_remove == 1 and p > 0 and (cur_o - 1) >= Y:
                    su_idx = 3
                else:
                    _, _, su_idx = max(candidates, key=lambda t: (t[0], t[1]))

                if su_idx == 1 and m > 0 and (cur_o - 2) >= Y:
                    m -= 1
                elif su_idx == 2 and n > 0 and (cur_o - 2) >= Y:
                    n -= 1
                elif su_idx == 3 and p > 0 and (cur_o - 1) >= Y:
                    p -= 1
                else:
                    break

            H_new[1] = m
            H_new[2] = n
            H_new[3] = p
            used_O_03 = int(n_0 + 2 * m + 2 * n + p)

        try:
            carbonyl_target_total = int(carbonyl_C)
            carbonyl_final_total = int(n_0 + m + n + p)
            self._carbonyl_shortage = max(0, int(carbonyl_target_total - carbonyl_final_total))
        except Exception:
            self._carbonyl_shortage = 0
        
        return H_new
    
    def _correct_oxygen_O(self, H: torch.Tensor, E_target: torch.Tensor) -> torch.Tensor:
        """
        修正O元素（28, 29号）
        """
        target_O = int(E_target[2].item())
        
        # 0,1,2,3号已被N修正和羰基修正处理
        n_0 = int(H[0].item())   # 0号 (1个O)
        n_1 = int(H[1].item())   # 1号 (2个O)
        n_2 = int(H[2].item())   # 2号 (2个O)
        n_3 = int(H[3].item())   # 3号 (1个O)
        
        # 0-3号已贡献的O
        used_O = n_0 * 1 + n_1 * 2 + n_2 * 2 + n_3 * 1
        
        # 需要修正的O数量
        W = target_O - used_O
        W = max(0, W)  # 防止负数
        
        if W == 0:
            return H
        
        H_new = H.clone()
        
        # 当前28、29号数量
        m = int(H[28].item())  # 28号 (OH)
        n = int(H[29].item())  # 29号 (-O-)
        
        current_total = m + n
        
        # 目标分布：28:29 = 0.55:0.45
        target_28 = W * 0.55
        target_29 = W * 0.45
        
        if current_total == W:
            return H_new
        
        if current_total < W:
            # 需要补充
            diff = W - current_total
            
            for _ in range(diff):
                delta_28 = m - target_28
                delta_29 = n - target_29
                
                # 补充负值最大的；偏差相同时优先补充优先级高的（28 > 29）
                if delta_28 < delta_29:
                    m += 1
                elif delta_29 < delta_28:
                    n += 1
                else:  # 偏差相同，优先补充28号
                    m += 1
        else:
            # 需要删除
            diff = current_total - W
            
            for _ in range(diff):
                delta_28 = m - target_28
                delta_29 = n - target_29
                
                # 删除正值最大的；偏差相同时优先删除优先级低的（29 < 28）
                if delta_28 > delta_29 and m > 0:
                    m -= 1
                elif delta_29 > delta_28 and n > 0:
                    n -= 1
                elif delta_28 == delta_29: 
                    if n > 0:
                        n -= 1
                    elif m > 0:
                        m -= 1
                elif m > 0:
                    m -= 1
                elif n > 0:
                    n -= 1
        H_new[28] = m
        H_new[29] = n
        
        return H_new
    
    def _correct_carbonyl_connection(self, H: torch.Tensor) -> torch.Tensor:
        """
        修正C=O连接（9号芳香羰基取代碳）
        """
        n_0 = int(H[0].item())
        n_1 = int(H[1].item())
        n_2 = int(H[2].item())
        n_3 = int(H[3].item())
        
        # 计算C=O总连接量
        W = n_0 * 1 + n_1 * 1 + n_2 * 1 + n_3 * 2
        target_9 = int(round(float(W) * 0.45))
        
        current_9 = int(H[9].item())
        
        H_new = H.clone()
        H_new[9] = target_9
        return H_new
    
    def _correct_ether_connection(self, H: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        修正-O-连接（5号芳香醚取代碳、19号醚接脂肪碳）
        
        Returns:
            Tuple[torch.Tensor, Dict[str, int]]: (修正后的直方图, 元数据字典)
        """
        n_2 = int(H[2].item())
        n_28 = int(H[28].item())
        n_29 = int(H[29].item())
        
        # 计算-O-总连接量
        W_ether = n_2 * 1 + n_28 * 1 + n_29 * 2
        
        m = int(H[5].item())   # 5号当前数量
        # 19号当前数量（可能包含之前其他修正的结果）
        n_current = int(H[19].item())

        
        H_new = H.clone()
        
        # 目标分布：5:19 = 0.8:0.2（仅O连接部分）
        target_5 = W_ether * 0.8
        target_19_ether = W_ether * 0.2
        
        current_total = int(m + n_current)
        
        if current_total == int(W_ether):
            m_target = int(m)
            n_19_target = int(n_current)
        elif current_total < int(W_ether):
            diff = int(W_ether) - int(current_total)
            m_target = int(m)
            n_19_target = int(n_current)
            for _ in range(diff):
                delta_5 = float(m_target) - float(target_5)
                delta_19 = float(n_19_target) - float(target_19_ether)
                if delta_5 < delta_19:
                    m_target += 1
                elif delta_19 < delta_5:
                    n_19_target += 1
                else:
                    m_target += 1
        else:
            diff = int(current_total) - int(W_ether)
            m_target = int(m)
            n_19_target = int(n_current)
            for _ in range(diff):
                delta_5 = float(m_target) - float(target_5)
                delta_19 = float(n_19_target) - float(target_19_ether)
                if delta_5 > delta_19 and int(m_target) > 0:
                    m_target -= 1
                elif delta_19 > delta_5 and int(n_19_target) > 0:
                    n_19_target -= 1
                else:
                    if int(n_19_target) > 0:
                        n_19_target -= 1
                    elif int(m_target) > 0:
                        m_target -= 1
        
        if int(m_target) < 0:
            m_target = 0
        if int(n_19_target) < 0:
            n_19_target = 0
        if int(m_target + n_19_target) != int(W_ether):
            n_19_target = max(0, int(W_ether) - int(m_target))
        
        H_new[5] = int(m_target)
        H_new[19] = int(n_19_target)  
        
        # 返回元数据供后续S连接修正使用
        meta = {'o_base_19': int(n_19_target)}
        
        return H_new, meta
    
    def _correct_thioether_connection(self, H: torch.Tensor, o_base_19: int = 0) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        修正-S-连接（7号芳香硫醚取代碳、19号硫醚接脂肪碳）
        
        Args:
            H: SU直方图
            o_base_19: O专用的19号基准（从_correct_ether_connection获取）
            
        Returns:
            Tuple[torch.Tensor, Dict[str, int]]: (修正后的直方图, 元数据字典)
        """
        n_31 = int(H[31].item())
        
        # 计算-S-总连接量
        W_thioether = n_31 * 2
        
        m = int(H[7].item())  # 7号当前数量

        H_new = H.clone()
        
        # 目标分布：7:19 = 0.4:0.6（仅S连接部分）
        target_7 = W_thioether * 0.4
        target_19_thioether = W_thioether * 0.6 
        
        # 计算S专用的19号（reserved）
        reserved_19 = max(0, W_thioether - m)
        
        # 重新分配7号和reserved_19
        n_19_thioether = 0  # S专用的19号（从0开始计算）
        
        current_total = m + n_19_thioether
        
        if current_total < W_thioether:
            # 需要补充
            diff = W_thioether - current_total
            
            for _ in range(diff):
                delta_7 = m - target_7
                delta_19 = n_19_thioether - target_19_thioether
                
                if delta_7 < delta_19:
                    m += 1
                elif delta_19 < delta_7:
                    n_19_thioether += 1
                else:  # 偏差相同，优先补充19号
                    n_19_thioether += 1
        else:
            # 需要删除（直接按目标分配）
            m = max(0, int(target_7))
            n_19_thioether = max(0, int(target_19_thioether))
            
            # 确保总和正确
            if m + n_19_thioether != W_thioether:
                n_19_thioether = W_thioether - m
        
        H_new[7] = m
        H_new[19] = o_base_19 + n_19_thioether
  
        # 返回元数据
        meta = {
            'o_base_19': o_base_19,
            's_reserved_19': n_19_thioether
        }
        
        return H_new, meta
    
    def _correct_amine_connection(self, H: torch.Tensor) -> torch.Tensor:
        """
        修正-NH-连接（6号芳香氨基取代碳、20号氨基接脂肪碳）
        """
        n_0 = int(H[0].item())
        n_27 = int(H[27].item())
        
        # 计算-NH-总连接量
        W = n_0 * 1 + n_27 * 2
        
        m = int(H[6].item())   # 6号当前数量
        n = int(H[20].item())  # 20号当前数量
        current_total = m + n

        H_new = H.clone()
        
        # 目标分布：6:20 = 0.6:0.4
        target_6 = W * 0.6
        target_20 = W * 0.4
        
        if current_total == W:
            return H_new
        
        if current_total < W:
            # 需要补充
            diff = W - current_total
            
            for _ in range(diff):
                delta_6 = m - target_6
                delta_20 = n - target_20
                
                # 补充负值最大的；偏差相同时优先补充优先级高的（6 > 20）
                if delta_6 < delta_20:
                    m += 1
                elif delta_20 < delta_6:
                    n += 1
                else:  # 偏差相同，优先补充6号
                    m += 1
        else:
            # 需要删除
            diff = current_total - W
            
            for _ in range(diff):
                delta_6 = m - target_6
                delta_20 = n - target_20
                
                # 删除正值最大的；偏差相同时优先删除优先级低的（20 < 6）
                if delta_6 > delta_20 and m > 0:
                    m -= 1
                elif delta_20 > delta_6 and n > 0:
                    n -= 1
                elif delta_6 == delta_20:  # 偏差相同，优先删除20号
                    if n > 0:
                        n -= 1
                    elif m > 0:
                        m -= 1
                elif m > 0:
                    m -= 1
                elif n > 0:
                    n -= 1
        
        H_new[6] = m
        H_new[20] = n
        
        return H_new
    
    def _correct_halogen_connection(self, H: torch.Tensor) -> torch.Tensor:
        """
        修正-X连接（8号芳香卤取代碳、21号卤接脂肪碳）
        """
        n_32 = int(H[32].item())
        
        # 计算-X总连接量
        W = n_32 * 1
        
        m = int(H[8].item())   
        n = int(H[21].item())  
        current_total = m + n
        
        H_new = H.clone()
        
        # 目标分布：8:21 = 0.6:0.4
        target_8 = W * 0.6
        target_21 = W * 0.4
        
        if current_total == W:
            return H_new
        
        if current_total < W:
            # 需要补充
            diff = W - current_total
            
            for _ in range(diff):
                delta_8 = m - target_8
                delta_21 = n - target_21
                
                # 补充负值最大的；偏差相同时优先补充优先级高的（8 > 21）
                if delta_8 < delta_21:
                    m += 1
                elif delta_21 < delta_8:
                    n += 1
                else:  # 偏差相同，优先补充8号
                    m += 1
        else:
            # 需要删除
            diff = current_total - W
            
            for _ in range(diff):
                delta_8 = m - target_8
                delta_21 = n - target_21
                
                # 删除正值最大的；偏差相同时优先删除优先级低的（21 < 8）
                if delta_8 > delta_21 and m > 0:
                    m -= 1
                elif delta_21 > delta_8 and n > 0:
                    n -= 1
                elif delta_8 == delta_21:  # 偏差相同，优先删除21号
                    if n > 0:
                        n -= 1
                    elif m > 0:
                        m -= 1
                elif m > 0:
                    m -= 1
                elif n > 0:
                    n -= 1
        
        H_new[8] = m
        H_new[21] = n
        return H_new
    

    def _correct_aliphatic_carbons(self, H: torch.Tensor, S_target: torch.Tensor,
                                   E_target: torch.Tensor) -> torch.Tensor:
        """
        修正脂肪碳结构（22, 23, 24, 25号）
        """
        budgets = self._estimate_region_carbon_budgets(S_target, E_target)
        M_float = float(budgets['aliphatic_C'])
        M = int(round(M_float))
        
        # 已用脂肪碳
        n_19 = int(H[19].item())
        n_20 = int(H[20].item())
        n_21 = int(H[21].item())
        
        if M == 0:
            return H
        
        H_new = H.clone()
        
        # 新规则：
        # 22 = 0.20M, 23 = 0.68M - 19 - 20 - 21, 24 = 0.10M, 25 = 0.02M
        target_22 = int(round(0.20 * M_float))
        target_24 = int(round(0.10 * M_float))
        target_25 = int(round(0.02 * M_float))
        target_23 = int(round(0.68 * M_float - float(n_19 + n_20 + n_21)))
        target_23 = max(0, target_23)
        
        H_new[22] = target_22
        H_new[23] = target_23
        H_new[24] = target_24
        H_new[25] = target_25 
        
        return H_new
    
    def _correct_unsaturated_carbons(self, H: torch.Tensor, S_target: torch.Tensor,
                                     E_target: torch.Tensor) -> torch.Tensor:
        """
        修正非饱和结构（14, 15, 16, 17, 18号）
        """
        budgets = self._estimate_region_carbon_budgets(S_target, E_target)
        aromatic_C = float(budgets['aromatic_C'])

        # 新规则：
        # W = 0.05 * aromatic_C，W 取最近偶数
        W = max(0, self._nearest_even_int(0.05 * float(aromatic_C)))
        
        if W == 0:
            return H
        
        H_new = H.clone()

        # 双键:三键 = 0.8:0.2，并要求两端都为偶数。
        double_bond = self._nearest_even_int(0.8 * float(W))
        double_bond = max(0, min(int(W), int(double_bond)))
        triple_bond = self._nearest_even_int(0.2 * float(W))
        triple_bond = max(0, int(triple_bond))

        # 若独立取偶数后不再守恒，则优先保持总W守恒。
        if int(double_bond + triple_bond) != int(W):
            triple_bond = max(0, int(W) - int(double_bond))
        if int(triple_bond) % 2 != 0:
            triple_bond = max(0, int(triple_bond) - 1)
            double_bond = max(0, int(W) - int(triple_bond))

        target_14, target_15, target_16 = self._allocate_ratio_counts(
            int(double_bond), (0.1, 0.65, 0.25)
        )

        # q:r = 0.5:0.5
        target_17 = int(triple_bond) // 2
        target_18 = int(triple_bond) // 2
        
        H_new[14] = target_14
        H_new[15] = target_15
        H_new[16] = target_16
        H_new[17] = target_17
        H_new[18] = target_18
        
        return H_new
    
    def _correct_aromatic_carbons(self, H: torch.Tensor, S_target: torch.Tensor,
                                  E_target: torch.Tensor) -> torch.Tensor:
        """
        修正芳香结构（10, 11, 12, 13号）
        """
        budgets = self._estimate_region_carbon_budgets(S_target, E_target)
        total_C = float(budgets['N'])
        xN = float(budgets['xN'])
        yN = float(budgets['yN'])
        aromatic_C = float(budgets['aromatic_C'])

        # 新规则：
        # W = 0.95 * aromatic_C - 4号
        fa = float(yN + 0.1 * xN) / max(1.0, float(total_C))
        n_4 = int(H[4].item())

        W_float = float(0.95 * float(aromatic_C) - float(n_4))
        W = max(0, int(round(W_float)))
        
        if W == 0:
            return H
        
        H_new = H.clone()
        
        n_5 = int(H[5].item())
        n_6 = int(H[6].item())
        n_7 = int(H[7].item())
        n_8 = int(H[8].item())
        n_9 = int(H[9].item())

        if fa <= 0.5:
            frac_10, frac_11, frac_12, frac_13 = 0.10, 0.22, 0.15, 0.53
        elif fa <= 0.6:
            frac_10, frac_11, frac_12, frac_13 = 0.10, 0.22, 0.18, 0.50
        elif fa <= 0.7:
            frac_10, frac_11, frac_12, frac_13 = 0.10, 0.20, 0.20, 0.50
        elif fa <= 0.75:
            frac_10, frac_11, frac_12, frac_13 = 0.09, 0.20, 0.225, 0.485
        elif fa <= 0.8:
            frac_10, frac_11, frac_12, frac_13 = 0.085, 0.19, 0.255, 0.47
        elif fa <= 0.85:
            frac_10, frac_11, frac_12, frac_13 = 0.08, 0.175, 0.285, 0.46
        elif fa <= 0.9:
            frac_10, frac_11, frac_12, frac_13 = 0.07, 0.17, 0.305, 0.445
        else:
            frac_10, frac_11, frac_12, frac_13 = 0.07, 0.165, 0.34, 0.425

        existing_5_9 = int(n_5 + n_6 + n_7 + n_8 + n_9)
        target_10 = int(round(float(frac_10) * float(W)))
        target_12 = int(round(float(frac_12) * float(W)))
        target_13 = int(round(float(frac_13) * float(W)))
        target_11 = int(round(float(frac_11) * float(W) - float(existing_5_9)))
        target_11 = max(0, target_11)

        aromatic_excess = int(existing_5_9 + target_10 + target_11 + target_12 + target_13 - W)
        if aromatic_excess > 0:
            reducible = min(int(target_13), int(aromatic_excess))
            target_13 -= reducible
            aromatic_excess -= reducible
        if aromatic_excess > 0:
            reducible = min(int(target_12), int(aromatic_excess))
            target_12 -= reducible
            aromatic_excess -= reducible
        if aromatic_excess > 0:
            reducible = min(int(target_10), int(aromatic_excess))
            target_10 -= reducible
            aromatic_excess -= reducible
        if aromatic_excess > 0:
            reducible = min(int(target_11), int(aromatic_excess))
            target_11 -= reducible
            aromatic_excess -= reducible

        aromatic_deficit = int(W - existing_5_9 - target_10 - target_11 - target_12 - target_13)
        if aromatic_deficit > 0:
            target_11 += int(aromatic_deficit)

        # 约束：10号数量必须为偶数；若为奇数，删除一个10号
        if int(target_10) % 2 != 0 and int(target_10) > 0:
            target_10 -= 1
        
        H_new[10] = target_10
        H_new[11] = target_11
        H_new[12] = target_12
        H_new[13] = target_13
        
        return H_new

    def _adjust_hydrogen(self, H: torch.Tensor, E_target: torch.Tensor) -> torch.Tensor:
        """
        H元素调整（三区域调整）
        """
        # 将H转移到CPU上以避免频繁的GPU同步
        H_cpu = H.cpu()
        E_SU_cpu = self.E_SU.cpu()
        
        E_current = torch.matmul(H_cpu.float(), E_SU_cpu)
        current_H = E_current[1].item()
        target_H = E_target[1].item()
        
        delta_H = current_H - target_H
        rel_error = abs(delta_H) / max(1.0, float(target_H))

        if rel_error < 0.03:
            return H
        
        W = abs(current_H - 1.02 * target_H) if delta_H > 0 else abs(current_H - 0.98 * target_H)
        W = int(W)
        if W <= 0:
            return H
        
        X = int(W * 0.4)
        Y = int(W * 0.3)
        Z = int(W * 0.3)

        H_new = H_cpu.clone()
        
        if delta_H > 0:
            H_new = self._reduce_hydrogen_aromatic(H_new, X)
            H_new = self._reduce_hydrogen_aliphatic(H_new, Y)
            H_new = self._reduce_hydrogen_unsaturated(H_new, Z)
        else:
            H_new = self._increase_hydrogen_aromatic(H_new, X)
            H_new = self._increase_hydrogen_aliphatic(H_new, Y)
            H_new = self._increase_hydrogen_unsaturated(H_new, Z)
            
        # 限制25号的数量不超过3%的脂肪碳总量！22号的数量不少于23号的10%！
        aliphatic_total = sum(H_new[i].item() for i in [19, 20, 21, 22, 23, 24, 25])
        max_25 = int(0.03 * aliphatic_total)
        if H_new[25] > max_25:
            diff = H_new[25] - max_25
            H_new[25] = max_25
            H_new[23] += diff
            
        min_22 = int(0.10 * H_new[23].item())
        if H_new[22] < min_22:
            diff = min_22 - H_new[22]
            H_new[22] = min_22
            H_new[23] -= diff
            if H_new[23] < 0:
                H_new[23] = 0

        # 返回时移回原来的设备
        return H_new.to(H.device)

    def _reduce_hydrogen_aromatic(self, H: torch.Tensor, X: int) -> torch.Tensor:
        """每次减少一个13号，轮流增加一个12号/12号/10号/12号/11号"""
        if X <= 0: return H
        H_new = H.clone()
        reduced = 0
        cycle = [12, 12, 11, 12, 12, 11, 10]
        c_idx = 0

        while reduced < X:
            if H_new[13] > 0:
                H_new[13] -= 1
                H_new[cycle[c_idx]] += 1
                reduced += 1
                c_idx = (c_idx + 1) % 4
            else:
                break
        return H_new

    def _reduce_hydrogen_aliphatic(self, H: torch.Tensor, Y: int) -> torch.Tensor:
        """第一轮减22增23/23/23/24，第二轮减23增24"""
        if Y <= 0: return H
        H_new = H.clone()
        reduced = 0
        
        step = 0
        stuck = 0
        
        while reduced < Y:
            if step < 4:
                # 第一轮: 减22, 增23/23/23/24
                if H_new[22] > 0:
                    tgt = 24 if step == 3 else 23
                    H_new[22] -= 1
                    H_new[tgt] += 1
                    diff = 3 - (1 if tgt == 24 else 2)
                    reduced += diff
                    stuck = 0
                else:
                    stuck += 1
                step += 1
            else:
                # 第二轮: 减23, 增24
                if H_new[23] > 0:
                    H_new[23] -= 1
                    H_new[24] += 1
                    reduced += 1
                    stuck = 0
                else:
                    stuck += 1
                step = 0
                
            if stuck > 5:
                break
        return H_new
    
    def _reduce_hydrogen_unsaturated(self, H: torch.Tensor, Z: int) -> torch.Tensor:
        """第一轮减16增15/14，第二轮减15增14，必须保证15 > 14"""
        if Z <= 0: return H
        H_new = H.clone()
        reduced = 0
        
        step = 0
        stuck = 0
        
        while reduced < Z:
            if step == 0:
                # 第一轮, 步1: -16, +15
                if H_new[16] > 0:
                    H_new[16] -= 1
                    H_new[15] += 1
                    reduced += 1
                    stuck = 0
                else:
                    stuck += 1
                step = 1
            elif step == 1:
                # 第一轮, 步2: -16, +14 (需保证 15 > 14)
                if H_new[16] > 0 and H_new[15] > (H_new[14] + 1):
                    H_new[16] -= 1
                    H_new[14] += 1
                    reduced += 2
                    stuck = 0
                else:
                    stuck += 1
                step = 2
            else:
                # 第二轮: -15, +14 (需保证 15 > 14)
                if H_new[15] > 0 and (H_new[15] - 1) > (H_new[14] + 1):
                    H_new[15] -= 1
                    H_new[14] += 1
                    reduced += 1
                    stuck = 0
                else:
                    stuck += 1
                step = 0
                
            if stuck > 3:
                break
        return H_new

    def _increase_hydrogen_aromatic(self, H: torch.Tensor, X: int) -> torch.Tensor:
        """每次增加一个13号，轮流减少一个11号/12号/10号"""
        if X <= 0: return H
        H_new = H.clone()
        increased = 0
        cycle = [11, 10, 12, 10]
        c_idx = 0
        stuck = 0
        
        while increased < X:
            tgt = cycle[c_idx]
            if H_new[tgt] > 0:
                H_new[tgt] -= 1
                H_new[13] += 1
                increased += 1
                stuck = 0
            else:
                stuck += 1
                if stuck >= 3:
                    break
            c_idx = (c_idx + 1) % 3
        return H_new

    def _increase_hydrogen_aliphatic(self, H: torch.Tensor, Y: int) -> torch.Tensor:
        """第一轮每次增加一个22号，轮流减少一个23/23/23/24/25；第二轮增加23，减少24"""
        if Y <= 0: return H
        H_new = H.clone()
        increased = 0
        
        r1_targets = [23, 23, 23, 24, 25]
        c_idx = 0
        stuck = 0
        
        while increased < Y:
            if c_idx < 5:
                tgt = r1_targets[c_idx]
                if H_new[tgt] > 0:
                    H_new[tgt] -= 1
                    H_new[22] += 1
                    diff = 3 - (2 if tgt == 23 else (1 if tgt == 24 else 0))
                    increased += diff
                    stuck = 0
                else:
                    stuck += 1
                c_idx += 1
            else:
                if H_new[24] > 0:
                    H_new[24] -= 1
                    H_new[23] += 1
                    increased += 1
                    stuck = 0
                else:
                    stuck += 1
                    
                if stuck > 6:
                    break
                c_idx = 0
        return H_new

    def _increase_hydrogen_unsaturated(self, H: torch.Tensor, Z: int) -> torch.Tensor:
        """第一轮每次增加16，减少15/14；第二轮增加15，减少14"""
        if Z <= 0: return H
        H_new = H.clone()
        increased = 0
        
        r1_targets = [15, 14]
        c_idx = 0
        stuck = 0
        
        while increased < Z:
            if c_idx < 2:
                tgt = r1_targets[c_idx]
                if H_new[tgt] > 0 and (H_new[16] + 1) <= (H_new[15] + H_new[14] - 1):
                    H_new[tgt] -= 1
                    H_new[16] += 1
                    diff = 2 - (1 if tgt == 15 else 0)
                    increased += diff
                    stuck = 0
                else:
                    stuck += 1
                c_idx += 1
            else:
                if H_new[14] > 0:
                    H_new[14] -= 1
                    H_new[15] += 1
                    increased += 1
                    stuck = 0
                else:
                    stuck += 1
                    
                if stuck > 4:
                    break
                c_idx = 0
        return H_new
    
