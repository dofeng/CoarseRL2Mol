import os
import math
import random
import copy
import io
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from contextlib import redirect_stdout
import torch

from .RL_state import (
    MCTSState,
    RigidStageEvaluator, FlexStageEvaluator,
    BranchStageEvaluator, SideStageEvaluator,
    SubstitutionStageEvaluator,
    compute_su_delta, find_overlapping_axials,
)
from .RL_allocator import FlexAllocator
from .RL_init import initialize
from .stage_rigid import RigidStage
from .stage_flex import FlexStage
from .stage_side import SideStage, mcts_state_to_raw_mol
from .stage_branch import BranchStage
from .state_substitutor import SubstitutionStage
from .utils import NMRPredictor, load_target_spectrum_from_csv
from .visualization import (
    save_rigid_beam_results,
    save_flex_beam_results,
    save_branch_beam_results,
    save_side_beam_results
)

@dataclass
class MCTSConfig:
    """MCTS configuration parameters.

    Effective parameters:
      - rigid: iterations, beam_width, max_cluster_size
      - flex: flex_iterations, flex_max_steps, flex_beam_width
      - branch/side/subst: branch_iterations, side_iterations, subst_iterations
      - stage-action pruning: branch_candidate_k, side_candidate_k, subst_candidate_k, stage_action_branching
      - NMR: nmr_model_ckpt, target_spectrum, spectrum_csv(alias), nmr_weight
      - misc: max_candidates, max_depth, max_aspect_ratio, seed
    """
    
    # UCT parameters
    c_puct: float = 1.414              
    
    # Search parameters
    iterations: int = 10             
    max_depth: int = 20              
    
    # Beam search parameters
    beam_width: int = 1               
    
    # Candidate limits
    max_candidates: int = 10          
    
    # Rigid stage parameters
    max_cluster_size: int = 4         
    
    # Flex stage parameters
    flex_iterations: int = 5       
    flex_max_steps: int = 10         
    flex_beam_width: int = 1          
    branch_iterations: int = 30
    side_iterations: int = 30
    subst_iterations: int = 12
    branch_candidate_k: int = 8
    side_candidate_k: int = 8
    subst_candidate_k: int = 6
    stage_action_branching: int = 2
    
    # Side stage parameters
    side_beam_width: int = 1          
    subst_n_variants: int = 3
    side_n_seeds: Optional[int] = None   # deprecated alias for subst_n_variants
    nmr_model_ckpt: Optional[str] = None   
    target_spectrum: Optional[str] = None
    # Backward-compatible alias used by older notebooks/scripts.
    spectrum_csv: Optional[str] = None
    nmr_weight: float = 20.0         
    
    # Geometric constraints
    max_aspect_ratio: float = 5.0      
    
    # Random seed
    seed: Optional[int] = None

    def __post_init__(self):
        # Accept both `target_spectrum` and legacy `spectrum_csv`.
        if self.target_spectrum is None and self.spectrum_csv is not None:
            self.target_spectrum = self.spectrum_csv
        elif self.spectrum_csv is None and self.target_spectrum is not None:
            self.spectrum_csv = self.target_spectrum
        if self.side_n_seeds is not None:
            self.subst_n_variants = int(self.side_n_seeds)

DEFAULT_CONFIG = MCTSConfig()

@dataclass
class BeamCandidate:
    """
    Beam search candidate - represents one possible result.
    """
    state: MCTSState                 
    score: float                      
    action_sequence: List[Dict]        
    info: Dict = field(default_factory=dict)  
    
    def __post_init__(self):
        if self.info is None:
            self.info = {}


@dataclass
class _StageBeamCandidate:
    stage: Any
    base_score: float
    stage_score: float
    action_sequence: List[Dict]
    info: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False

    @property
    def score(self) -> float:
        return float(self.base_score) + float(self.stage_score)


def _normalize_stage_action(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, tuple):
        return tuple(_normalize_stage_action(x) for x in obj)
    if isinstance(obj, list):
        return tuple(_normalize_stage_action(x) for x in obj)
    if isinstance(obj, dict):
        items = []
        for k, v in sorted(obj.items(), key=lambda kv: str(kv[0])):
            if str(k) in {'score', 'spatial_score', 'prior_score'}:
                continue
            items.append((str(k), _normalize_stage_action(v)))
        return tuple(items)
    if hasattr(obj, 'chain_type') and hasattr(obj, 'composition'):
        return (
            'ChainSpec',
            str(getattr(obj, 'chain_type', '')),
            tuple(int(x) for x in list(getattr(obj, 'composition', []) or [])),
            str(getattr(obj, 'origin_type', '')),
            tuple(int(x) for x in list(getattr(obj, 'source_ids', []) or [])),
        )
    if hasattr(obj, 'id') and hasattr(obj, 'kind'):
        return ('Cluster', int(getattr(obj, 'id', -1)), str(getattr(obj, 'kind', '')))
    return repr(obj)


@dataclass
class StageSearchNode:
    stage: Any
    action: Optional[Dict[str, Any]] = None
    parent: Optional['StageSearchNode'] = None
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[str, 'StageSearchNode'] = field(default_factory=dict)
    _untried_actions: Optional[List[Dict[str, Any]]] = None

    def get_untried_actions(self, candidate_k: int) -> List[Dict[str, Any]]:
        if self._untried_actions is None:
            self._untried_actions = list(self.stage.get_candidates(k=int(candidate_k)) or [])
        return self._untried_actions

    def is_terminal(self) -> bool:
        return bool(self.stage.is_done())

    def is_fully_expanded(self, candidate_k: int) -> bool:
        return len(self.get_untried_actions(candidate_k)) == 0

    def best_child(self, c_puct: float) -> Optional['StageSearchNode']:
        best_score = -float('inf')
        best_child = None
        for child in self.children.values():
            if child.visits <= 0:
                score = float('inf')
            else:
                exploit = child.value_sum / max(1, child.visits)
                explore = c_puct * child.prior * math.sqrt(max(self.visits, 1)) / (1 + child.visits)
                score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child


class StageSearchEngine:
    """Unified MCTS engine shared across flex / branch / side / substitution."""

    def __init__(self,
                 iterations: int = 32,
                 c_puct: float = 1.414,
                 candidate_k: int = 12,
                 rollout_depth: int = 3):
        self.iterations = max(1, int(iterations))
        self.c_puct = float(c_puct)
        self.candidate_k = max(1, int(candidate_k))
        self.rollout_depth = max(1, int(rollout_depth))

    def search_actions(self,
                       stage: Any,
                       evaluator: Any,
                       branch_width: int = 1,
                       candidate_k: Optional[int] = None) -> List[Dict[str, Any]]:
        candidate_k = max(1, int(candidate_k or self.candidate_k))
        branch_width = max(1, int(branch_width))

        root = StageSearchNode(stage=stage.clone())
        root.get_untried_actions(candidate_k)
        if not root._untried_actions:
            return []

        for _ in range(self.iterations):
            self._iterate(root, evaluator, candidate_k)

        ranked = sorted(
            root.children.values(),
            key=lambda child: (
                child.visits,
                child.value_sum / max(1, child.visits),
                child.prior,
            ),
            reverse=True,
        )
        return [copy.deepcopy(child.action) for child in ranked[:branch_width] if child.action is not None]

    def _iterate(self, root: StageSearchNode, evaluator: Any, candidate_k: int) -> None:
        node = root

        while not node.is_terminal() and node.is_fully_expanded(candidate_k):
            child = node.best_child(self.c_puct)
            if child is None:
                break
            node = child

        if not node.is_terminal():
            expanded = self._expand(node, evaluator, candidate_k)
            if expanded is not None:
                node = expanded

        value = self._rollout(node.stage.clone(), evaluator, candidate_k)

        while node is not None:
            node.visits += 1
            node.value_sum += float(value)
            node = node.parent

    def _expand(self,
                node: StageSearchNode,
                evaluator: Any,
                candidate_k: int) -> Optional[StageSearchNode]:
        actions = node.get_untried_actions(candidate_k)
        while actions:
            action = actions.pop(0)
            next_stage = node.stage.clone()
            if not next_stage.step(copy.deepcopy(action)):
                continue
            prior = float(evaluator.prior(next_stage, action))
            child = StageSearchNode(
                stage=next_stage,
                action=copy.deepcopy(action),
                parent=node,
                prior=prior,
            )
            node.children[repr(_normalize_stage_action(action))] = child
            return child
        return None

    def _rollout(self, stage: Any, evaluator: Any, candidate_k: int) -> float:
        for _ in range(self.rollout_depth):
            if stage.is_done():
                break
            candidates = list(stage.get_candidates(k=candidate_k) or [])
            if not candidates:
                break
            action = evaluator.select_rollout_action(stage, candidates)
            if action is None:
                break
            next_stage = stage.clone()
            if not next_stage.step(copy.deepcopy(action)):
                fallback_ok = False
                for alt in candidates[1:]:
                    next_stage = stage.clone()
                    if next_stage.step(copy.deepcopy(alt)):
                        fallback_ok = True
                        break
                if not fallback_ok:
                    break
            stage = next_stage
        return float(evaluator.evaluate(stage))

# ==================== MCTS Node ====================
class MCTSNode:
    """MCTS tree node"""
    
    def __init__(
        self,
        state: MCTSState,
        action: Optional[Dict] = None,
        parent: Optional['MCTSNode'] = None,
    ):
        self.state = state
        self.action = action
        self.parent = parent
        self.children: Dict[str, 'MCTSNode'] = {}
        
        self.visits: int = 0
        self.value: float = 0.0
        self.prior: float = 0.0  
        
        self._untried_actions: Optional[List[Dict]] = None
        self._is_terminal: Optional[bool] = None
    
    def is_fully_expanded(self, stage) -> bool:
        """Check if all actions have been tried"""
        if self._untried_actions is None:
            self._untried_actions = stage.get_candidates()
        return len(self._untried_actions) == 0
    
    def is_terminal(self, stage) -> bool:
        """Check if node is terminal (stage done)"""
        if self._is_terminal is None:
            self._is_terminal = stage.is_done()
        return self._is_terminal
    
    def best_child(self, c_puct: float) -> 'MCTSNode':
        """Select best child using UCT formula"""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children.values():
            if child.visits == 0:
                score = float('inf')
            else:
                exploit = child.value / child.visits
                explore = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
                score = exploit + explore
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, stage) -> 'MCTSNode':
        """Expand one untried action. Skip actions whose step() fails.
        Works with any stage type that has step() and get_candidates() methods.
        """
        if self._untried_actions is None:
            self._untried_actions = stage.get_candidates()
        
        # Try actions until one succeeds or all exhausted
        while self._untried_actions:
            action = self._untried_actions.pop(0)
            
            new_state = self.state.copy()
            # Reconstruct stage from the new state (supports RigidStage)
            new_stage = RigidStage(new_state, skip_init=True,
                                   max_cluster_size=getattr(stage, 'max_cluster_size', 4))
            
            if not new_stage.step(action):
                continue  # Step failed — skip this action
            
            child = MCTSNode(
                state=new_state,
                action=action,
                parent=self,
            )
            
            raw = action.get('score', 0)
            child.prior = 1.0 / (1.0 + math.exp(raw / 100.0))
            
            key = _action_key(action)
            self.children[key] = child
            return child
        
        return self  # All actions exhausted
    
    def best_action(self) -> Optional[Dict]:
        """Get best action based on visit counts"""
        if not self.children:
            return None
        
        best_child = max(self.children.values(), key=lambda c: c.visits)
        return best_child.action


def _action_key(action: Dict) -> str:
    """Generate unique key for action"""
    if action is None:
        return "root"
    
    atype = action.get('type', 'unknown')
    if atype == 'rigid':
        return f"rigid_{action['cluster_a_id']}_{action['cluster_b_id']}_{action['site_a_idx']}_{action['site_b_idx']}"
    elif atype == 'flex':
        return (
            f"flex_{action.get('bridge_idx', -1)}_{action['cluster_a_id']}_{action['cluster_b_id']}_"
            f"{action['site_a_idx']}_{action['site_b_idx']}_{action['chain_len']}"
        )
    return f"unknown_{id(action)}"
# ==================== Reward Calculator ====================

class RewardCalculator:
    """Calculate rewards for MCTS"""
    def __init__(self, config: MCTSConfig):
        self.config = config
    
    def score_rigid(self, _action: Dict, state: MCTSState) -> float:
        """
        Reward for a rigid connection action.
        """
        score = 0.0
        
        # SU 10 consumption reward (0 to 0.4)
        remaining = state.su_counts.get(10, 0)
        total_sites = sum(1 for c in state.graph.clusters
                          for s in c.sites if s.su_type in (10, 13))
        total_10 = total_sites + len(state.graph.rigid) * 2
        if total_10 > 0:
            consumed_frac = 1.0 - remaining / max(total_10, 1)
            score += 0.4 * consumed_frac
        
        # Coverage reward: fraction of RCs with rigid edges (0 to 0.4)
        stage = RigidStage(state, skip_init=True)
        rc_map = stage.get_rc_map()
        if rc_map:
            n_with_edges = sum(1 for r in rc_map
                               if stage._rc_has_rigid_edges(r, rc_map))
            score += 0.4 * (n_with_edges / len(rc_map))
        
        # Compactness (0 to 0.2)
        aspect = state.get_aspect_ratio()
        if aspect <= self.config.max_aspect_ratio:
            score += 0.2 * (1.0 / (1.0 + abs(aspect - 1.0) * 0.3))
        
        return score
    
    def score_terminal(self, state: MCTSState) -> float:
        """Calculate terminal state reward (higher = better)."""
        score = 0.0
        
        # Primary: SU 10 consumption (0 to 0.5)
        remaining = state.su_counts.get(10, 0)
        if remaining < 2:
            score += 0.5  # All consumed
        else:
            score += 0.3 * (1.0 - remaining / 50.0)
        
        # Coverage bonus (0 to 0.3)
        stage = RigidStage(state, skip_init=True)
        rc_map = stage.get_rc_map()
        if rc_map:
            n_with_edges = sum(1 for r in rc_map
                               if stage._rc_has_rigid_edges(r, rc_map))
            coverage = n_with_edges / len(rc_map)
            score += 0.3 * coverage
        
        # Giant RC penalty (0 to -0.2)
        if rc_map:
            sizes = [stage._rc_ring_count(r, rc_map) for r in rc_map]
            max_size = max(sizes) if sizes else 0
            if max_size > self.config.max_cluster_size * 2:
                score -= 0.2
        
        # Aspect ratio (0 to 0.1)
        aspect = state.get_aspect_ratio()
        if aspect <= self.config.max_aspect_ratio:
            score += 0.1
        
        return score

# ==================== Beam MCTS Search ====================
class BeamMCTSSearch:
    """
    Beam MCTS search algorithm.
    """
    
    def __init__(self, config: MCTSConfig = None, nodes_csv: Optional[str] = None):
        self.config = config or DEFAULT_CONFIG
        self.nodes_csv = nodes_csv
        self.reward_calc = RewardCalculator(self.config)
        
        if self.config.seed is not None:
            random.seed(self.config.seed)
    
    def search(self, initial_state: MCTSState) -> List[BeamCandidate]:
        """
        Run Beam MCTS search for rigid stage.
        """
        candidates = [BeamCandidate(
            state=initial_state.copy(),
            score=0.0,
            action_sequence=[],
            info={'stage': 'rigid'}
        )]
        
        max_steps = 80  # Safety limit (typical: ~15 connections for 30 SU 10)
        
        for step in range(1, max_steps + 1):
            new_candidates = []
            any_active = False
            
            for cand in candidates:
                if cand.info.get('completed'):
                    new_candidates.append(cand)
                    continue
                
                # Check if this candidate is done
                stage = RigidStage(
                    cand.state,
                    skip_init=True,
                    max_cluster_size=self.config.max_cluster_size,
                )
                
                if stage.is_done():
                    cand.info['completed'] = True
                    new_candidates.append(cand)
                    continue
                
                any_active = True
                
                # Run MCTS iterations to explore connection choices
                root = MCTSNode(cand.state.copy())
                for _ in range(self.config.iterations):
                    self._iterate(root)
                
                # Extract top-K diverse actions from MCTS tree
                top_actions = self._get_diverse_actions(
                    root, k=self.config.beam_width)
                
                if not top_actions:
                    # No valid actions found — try direct greedy fallback
                    direct_cands = stage.get_candidates(k=5)
                    top_actions = direct_cands[:self.config.beam_width]
                
                if not top_actions:
                    cand.info['completed'] = True
                    new_candidates.append(cand)
                    continue
                
                # Expand beam: try each action
                expanded = False
                for action in top_actions:
                    new_state = cand.state.copy()
                    new_stage = RigidStage(
                        new_state,
                        skip_init=True,
                        max_cluster_size=self.config.max_cluster_size,
                    )
                    
                    if new_stage.step(action):
                        reward = self.reward_calc.score_rigid(action, new_state)
                        new_candidates.append(BeamCandidate(
                            state=new_state,
                            score=cand.score + reward,
                            action_sequence=cand.action_sequence + [action],
                            info={'stage': 'rigid', 'step': step},
                        ))
                        expanded = True
                
                if not expanded:
                    # MCTS actions all failed — try broader direct fallback
                    direct_all = stage.get_candidates(k=20)
                    for action in direct_all:
                        new_state = cand.state.copy()
                        new_stage = RigidStage(
                            new_state,
                            skip_init=True,
                            max_cluster_size=self.config.max_cluster_size,
                        )
                        if new_stage.step(action):
                            reward = self.reward_calc.score_rigid(action, new_state)
                            new_candidates.append(BeamCandidate(
                                state=new_state,
                                score=cand.score + reward,
                                action_sequence=cand.action_sequence + [action],
                                info={'stage': 'rigid', 'step': step},
                            ))
                            expanded = True
                            break  # One success is enough for fallback
                
                if not expanded:
                    # Truly stuck — mark completed
                    cand.info['completed'] = True
                    new_candidates.append(cand)
            
            if not any_active or not new_candidates:
                break
            
            # Prune beam
            candidates = self._select_diverse_candidates(new_candidates)
            
            # Check if all done
            if all(c.info.get('completed', False) for c in candidates):
                break
            
            # Progress log
            if step % 3 == 0:
                active = sum(1 for c in candidates
                             if not c.info.get('completed'))
                su10_range = [c.state.su_counts.get(10, 0) for c in candidates
                              if not c.info.get('completed')]
                su10_str = (f"SU10={min(su10_range)}-{max(su10_range)}"
                            if su10_range else "all done")
                print(f"[BeamMCTS] Step {step}: {len(candidates)} cands "
                      f"({active} active) {su10_str}")
        
        # Final scoring: reward coverage and SU 10 exhaustion
        for cand in candidates:
            stage = RigidStage(cand.state, skip_init=True)
            rc_map = stage.get_rc_map()
            dist = stage.get_rigid_cluster_distribution()
            cand.info['rigid_clusters'] = dist
            cand.info['num_rigid_clusters'] = len(dist)
            
            # Coverage bonus: more RCs with edges is better
            n_with = sum(1 for d in dist if d.get('has_rigid_edges'))
            cand.score += n_with * 0.15
            
            # SU 10 exhaustion bonus
            remaining = cand.state.su_counts.get(10, 0)
            if remaining < 2:
                cand.score += 1.0
            else:
                cand.score -= remaining * 0.05

            flex_potential = self._estimate_flex_potential(cand.state)
            cand.info['flex_potential'] = flex_potential
            cand.score += flex_potential
        
        candidates.sort(key=lambda c: -c.score)
        return candidates[:self.config.beam_width]
    
    def _iterate(self, root: MCTSNode):
        """Run one MCTS iteration (Select → Expand → Simulate → Backprop)"""
        node = root
        temp_stage = RigidStage(
            node.state.copy(),
            skip_init=True,
            max_cluster_size=self.config.max_cluster_size,
        )
        
        # Selection — descend tree via UCT
        while not node.is_terminal(temp_stage) and node.is_fully_expanded(temp_stage):
            child = node.best_child(self.config.c_puct)
            if child is None:
                break
            node = child
            temp_stage = RigidStage(
                node.state.copy(), skip_init=True,
                max_cluster_size=self.config.max_cluster_size,
            )
        
        # Expansion — try to add a new child
        if not node.is_terminal(temp_stage):
            expanded = node.expand(temp_stage)
            if expanded is not node:  # expand succeeded
                node = expanded
        
        # Simulation (rollout)
        value = self._simulate(node.state.copy())
        
        # Backpropagation
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def _simulate(self, state: MCTSState, depth: int = 0) -> float:
        """Simulate random rollout — keep connecting until done or max depth."""
        stage = RigidStage(
            state,
            skip_init=True,
            max_cluster_size=self.config.max_cluster_size,
        )
        
        if depth >= self.config.max_depth or stage.is_done():
            return self.reward_calc.score_terminal(state)
        
        candidates = stage.get_candidates(k=10)
        if not candidates:
            return self.reward_calc.score_terminal(state)
        
        # Weighted random selection (lower score = better → higher weight)
        max_score = max(c.get('score', 0) for c in candidates)
        weights = [max_score - c.get('score', 0) + 1 for c in candidates]
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
            action = random.choices(candidates, weights=weights, k=1)[0]
        else:
            action = random.choice(candidates)
        
        if not stage.step(action):
            # Step failed — try next best
            for alt in candidates[1:5]:
                if stage.step(alt):
                    break
            else:
                return self.reward_calc.score_terminal(state)
        
        return self._simulate(state, depth + 1)
    
    def _get_diverse_actions(self, root: MCTSNode, k: int) -> List[Dict]:
        """Get top-K diverse actions from MCTS tree."""
        if not root.children:
            return []
        
        # Sort children by visit count
        sorted_children = sorted(
            root.children.values(), 
            key=lambda n: n.visits, 
            reverse=True
        )
        
        actions = []
        seen_clusters = set()
        
        for child in sorted_children:
            if len(actions) >= k:
                break
            
            action = child.action
            if action is None:
                continue
            
            # Diversity: prefer actions involving different clusters
            cluster_pair = (action.get('cluster_a_id'), action.get('cluster_b_id'))
            if cluster_pair not in seen_clusters:
                actions.append(action)
                seen_clusters.add(cluster_pair)
        
        return actions
    
    def _select_diverse_candidates(self, candidates: List[BeamCandidate]) -> List[BeamCandidate]:
        """Select diverse candidates based on score and structural uniqueness.

        Unlike the legacy implementation, this strictly respects beam_width.
        """
        if not candidates:
            return []
        
        # Sort by score (best first)
        candidates.sort(key=lambda c: -c.score)
        
        # Structural deduplication: use (n_rigid_edges, n_components, last_action_key) as fingerprint
        selected = []
        seen_fingerprints = set()
        
        for cand in candidates:
            n_rigid = len(cand.state.graph.rigid)
            n_comp = cand.state.get_component_count()
            last_key = _action_key(cand.action_sequence[-1]) if cand.action_sequence else ''
            fp = (n_rigid, n_comp, last_key)
            
            if fp not in seen_fingerprints:
                selected.append(cand)
                seen_fingerprints.add(fp)
            elif len(selected) < self.config.beam_width:
                # Allow duplicates if we don't have enough diversity
                selected.append(cand)
            
            if len(selected) >= self.config.beam_width:
                break
        
        return selected

    def _estimate_flex_potential(self, state: MCTSState) -> float:
        """Lightweight rigid→flex compatibility score.
        Higher means the rigid scaffold is more likely to admit valid flex growth.
        """
        if not self.nodes_csv:
            return 0.0
        try:
            with redirect_stdout(io.StringIO()):
                sim_state = state.copy()
                rigid_stage = RigidStage(
                    sim_state,
                    skip_init=True,
                    max_cluster_size=self.config.max_cluster_size,
                )
                rigid_stage.place_all_remaining()

                allocator = FlexAllocator(self.nodes_csv)
                alloc_result = allocator.allocate()
                flex_stage = FlexStage(
                    sim_state,
                    sim_state.su_counts.copy(),
                    allocation_result=alloc_result,
                )

                candidates = flex_stage.get_candidates(k=min(12, self.config.max_candidates))
                if not candidates:
                    return -0.8 if not flex_stage._all_clusters_placed_and_connected() else 0.2

                cross = sum(
                    1 for c in candidates
                    if sim_state._find(c['cluster_a_id']) != sim_state._find(c['cluster_b_id'])
                )
                unique_pairs = len({
                    (c['cluster_a_id'], c['cluster_b_id'], c.get('direction'))
                    for c in candidates
                })
                score = min(len(candidates), 8) * 0.10 + cross * 0.12 + unique_pairs * 0.04

                top_action = candidates[0]
                if flex_stage.step(top_action):
                    score += 0.35
                    score += flex_stage._bridges_done * 0.12
                    follow_up = flex_stage.get_candidates(k=8)
                    score += min(len(follow_up), 6) * 0.05
                    score -= max(sim_state.get_component_count() - 1, 0) * 0.03

                return score
        except Exception:
            return 0.0


# ==================== Main Runner ====================

class MCTSRunner:
    """
    MCTS runner for rigid stage.
    """
    
    def __init__(
        self,
        nodes_csv: str,
        spectrum_csv: Optional[str] = None,
        config: MCTSConfig = None,
    ):
        """
        Initialize MCTS runner.
        
        Args:
            nodes_csv: Path to nodes CSV file
            spectrum_csv: Optional path to NMR spectrum CSV
            config: MCTS configuration
        """
        self.nodes_csv = nodes_csv
        self.config = config or DEFAULT_CONFIG
        # Prefer explicit arg, otherwise fall back to config aliases.
        if spectrum_csv is None:
            spectrum_csv = self.config.target_spectrum or self.config.spectrum_csv
        self.spectrum_csv = spectrum_csv
        if self.config.target_spectrum is None and spectrum_csv is not None:
            self.config.target_spectrum = spectrum_csv
        if self.config.spectrum_csv is None and spectrum_csv is not None:
            self.config.spectrum_csv = spectrum_csv
        # Initialize from RL_init (generates aromatic clusters)
        self.init_data = initialize(nodes_csv, spectrum_csv)
        
        # Create initial state with clusters from initialization
        self.initial_state = self._create_initial_state()
    
    def _create_initial_state(self) -> MCTSState:
        """Create initial MCTS state from initialization data"""
        return MCTSState(
            graph=self.init_data['graph'],
            su_counts=self.init_data['su_counts'].copy(),
            reserved_su={},
            stage='rigid',
            step_count=0,
        )
    
    def run_rigid_stage(self, output_dir: Optional[str] = None) -> Dict:
        """
        Run Beam MCTS search for rigid connection stage.
        """
        print(f"\n[RigidStage] Starting (beam={self.config.beam_width}, iter={self.config.iterations})")
        
        # Run Beam MCTS search
        mcts = BeamMCTSSearch(self.config, nodes_csv=self.nodes_csv)
        beam_candidates = mcts.search(self.initial_state.copy())
        
        # Place ALL remaining clusters and get summaries
        summaries = []
        for i, cand in enumerate(beam_candidates):
            stage = RigidStage(cand.state, skip_init=True)
            greedy_added = stage.consume_all_possible_connections()
            # CRITICAL: Place all unplaced clusters (from original mcts_controller.py)
            # After rigid stage, ALL clusters must have positions for next stage
            stage.place_all_remaining()
            greedy_added += stage.consume_all_possible_connections()
            remaining_su10 = int(stage.state.su_counts.get(10, 0))
            if remaining_su10 != 0:
                raise RuntimeError(
                    f"Rigid stage failed to exhaust all SU10 connections for candidate #{i+1}: "
                    f"remaining_su10={remaining_su10}"
                )
            cand.state = stage.state
            summary = stage.get_result()
            summary['candidate_index'] = i
            summary['score'] = cand.score
            summary['greedy_rigid_added'] = int(greedy_added)
            summaries.append(summary)
        
        result = {
            'beam_candidates': beam_candidates,
            'best_candidate': beam_candidates[0] if beam_candidates else None,
            'summaries': summaries,
        }
        
        # Print concise results
        print(f"[RigidStage] Complete: {len(beam_candidates)} candidates")
        for i, (cand, summary) in enumerate(zip(beam_candidates, summaries)):
            num_rc = summary.get('rigid_cluster_count', 0)
            print(f"  #{i+1}: {summary['rigid_edges']} edges, {summary['total_clusters']} clusters, "
                  f"{num_rc} rigid-clusters, score={cand.score:.2f}, remain10={summary.get('remaining_su10', 0)}")
        
        # Visualize results if output_dir provided
        if output_dir:
            self._visualize_beam_results(beam_candidates, summaries, output_dir)
        
        return result

    def run_flex_stage(self, rigid_result: Dict, output_dir: Optional[str] = None) -> Dict:
        """
        Run flex stage on top of rigid stage results using the unified
        stage-search engine. The resource forms still come directly from
        FlexAllocator; search only chooses valid placement sites/actions.
        """
        print(f"\n[FlexStage] Starting (beam={self.config.flex_beam_width}, iter={self._stage_iterations('flex')})")
        
        rigid_candidates = rigid_result.get('beam_candidates', [])
        if not rigid_candidates:
            print("[FlexStage] No rigid candidates to build on!")
            return {'beam_candidates': [], 'best_candidate': None, 'summaries': []}

        raw_result = self._run_stage_with_engine('flex', rigid_candidates)
        flex_candidates = list(raw_result.get('beam_candidates', []))
        if not flex_candidates:
            print("[FlexStage] No flex candidates produced!")
            return raw_result

        complete_flex_candidates = [
            cand for cand in flex_candidates
            if bool((cand.info or {}).get('flex_result', {}).get('all_bridges_placed', False))
        ]
        if complete_flex_candidates:
            flex_candidates = complete_flex_candidates
        else:
            connected_flex_candidates = [
                cand for cand in flex_candidates
                if bool((cand.info or {}).get('flex_result', {}).get('all_connected', False))
            ]
            if connected_flex_candidates:
                print("[FlexStage] No candidate placed all allocated bridges; accepting connected partial flex results and redistributing remaining flex 23.")
                partial_pool = connected_flex_candidates
            else:
                print("[FlexStage] No candidate placed all allocated bridges or fully connected the scaffold; forcing flex stop, finalizing partial layouts, and redistributing remaining flex 23.")
                partial_pool = list(flex_candidates)

            flex_candidates = [
                self._finalize_partial_flex_candidate(cand)
                for cand in partial_pool
            ]
            flex_candidates.sort(key=self._flex_result_rank_key, reverse=True)

        flex_candidates = flex_candidates[:self.config.flex_beam_width]
        
        # Print concise summary
        print(f"[FlexStage] Complete: {len(flex_candidates)} candidates")
        for i, cand in enumerate(flex_candidates):
            fr = cand.info.get('flex_result', {})
            redist = fr.get('flex_23_redistribution', {}) or {}
            extra = ""
            if redist.get('applied_23', 0) > 0:
                extra = (f", reallocated23={redist.get('applied_23',0)}"
                         f"(side={redist.get('to_side',0)}, branch={redist.get('to_branch',0)})")
            print(f"  #{i+1}: {fr.get('connections_made',0)}/{fr.get('bridges_total',0)} bridges, "
                  f"connected={fr.get('all_connected',False)}, score={cand.score:.2f}{extra}")
        
        result = {
            'beam_candidates': flex_candidates,
            'best_candidate': flex_candidates[0] if flex_candidates else None,
            'summaries': [c.info.get('flex_result', {}) for c in flex_candidates],
        }
        
        # Visualize
        if output_dir:
            self._visualize_flex_results(flex_candidates, result['summaries'], output_dir)
        
        return result

    def _flex_result_rank_key(self, cand: BeamCandidate) -> tuple:
        fr = cand.info.get('flex_result', {})
        return (
            1 if fr.get('all_bridges_placed', False) else 0,
            1 if fr.get('all_connected', False) else 0,
            fr.get('bridges_done', 0),
            fr.get('connections_made', 0),
            cand.score,
        )

    def _reallocate_failed_flex_23(self, cand: BeamCandidate) -> BeamCandidate:
        info = cand.info or {}
        fr = dict(info.get('flex_result', {}) or {})
        allocator = info.get('allocator')
        if allocator is None:
            cand.info['flex_result'] = fr
            return cand

        remaining_flex_23 = int(fr.get('remaining_flex_23', 0))
        summary = allocator.redistribute_remaining_flex_23(remaining_flex_23)
        fr['flex_23_redistribution'] = summary
        fr['remaining_flex_23'] = int(summary.get('remaining_23', 0))
        fr['redistributed_flex_23'] = int(summary.get('applied_23', 0))
        fr['redistributed_to_side'] = int(summary.get('to_side', 0))
        fr['redistributed_to_branch'] = int(summary.get('to_branch', 0))
        fr['bridges_reallocated_to_side_branch'] = bool(summary.get('applied_23', 0) > 0)
        cand.info['flex_result'] = fr
        try:
            cand.info['allocation_result'] = copy.deepcopy(allocator._result)
        except Exception:
            pass
        return cand

    def _finalize_partial_flex_candidate(self, cand: BeamCandidate) -> BeamCandidate:
        """Accept a partial flex result and make it usable for later stages.

        This is the hard stop fallback: once flex search hits its limits, we
        stop trying to place more bridge chains, put any still-unplaced rigid
        components back onto the canvas, and pour the remaining flex-body 23s
        into side/branch pools.
        """
        info = cand.info or {}
        fr = dict(info.get('flex_result', {}) or {})

        stage = RigidStage(
            cand.state,
            skip_init=True,
            max_cluster_size=self.config.max_cluster_size,
        )
        n_unplaced_before = sum(1 for c in cand.state.graph.clusters if not c.placed)
        if n_unplaced_before > 0:
            stage.place_all_remaining()
        fr['partial_flex_accepted'] = True
        fr['unplaced_clusters_before_finalize'] = int(n_unplaced_before)
        fr['all_clusters_placed_after_finalize'] = all(c.placed for c in cand.state.graph.clusters)
        fr['all_connected'] = cand.state.is_all_connected()
        fr['components'] = cand.state.get_component_count()
        fr['aspect_ratio'] = cand.state.get_aspect_ratio()
        cand.info['flex_result'] = fr

        cand = self._reallocate_failed_flex_23(cand)
        return cand

    def _subst_result_rank_key(self, cand: BeamCandidate) -> tuple:
        info = cand.info or {}
        return (
            1 if info.get('subst_complete', False) else 0,
            -int(info.get('subst_remaining_total', 10**9)),
            -int(info.get('subst_l1_delta', 10**9)),
            float(info.get('nmr_score', 0.0)),
            float(cand.score),
        )

    def run_side_stage(self, branch_result: Dict, output_dir: Optional[str] = None) -> Dict:
        """
        Run side stage using the unified stage-search engine.
        """
        print(f"\n[SideStage] Starting (beam={self.config.side_beam_width}, iter={self._stage_iterations('side')})")
        
        branch_candidates = branch_result.get('beam_candidates', [])
        if not branch_candidates:
            print("[SideStage] No branch candidates to build on!")
            return {'beam_candidates': [], 'best_candidate': None, 'summaries': []}
        result = self._run_stage_with_engine('side', branch_candidates)
        side_candidates = list(result.get('beam_candidates', []))
        if not side_candidates:
            print("[SideStage] No candidates produced!")
            return result
        
        print(f"[SideStage] Complete: {len(side_candidates)} candidates")
        for i, c in enumerate(side_candidates[:3]):
            sr = c.info.get('side_result', {})
            nmr_str = f", NMR={c.info.get('nmr_score', 0.0):.3f}" if 'nmr_score' in c.info else ""
            print(f"  #{i+1}: {sr.get('sides_placed',0)}/{sr.get('sides_total',0)} sides, score={c.score:.2f}{nmr_str}")

        if output_dir:
            self._visualize_side_results(side_candidates, result.get('summaries', []), output_dir)
            
        return result
    
    def _visualize_beam_results(self, candidates: List[BeamCandidate], summaries: List[Dict], output_dir: str):
        """Visualize all beam candidates."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            save_rigid_beam_results(candidates, summaries, output_dir)
            print(f"\n[MCTSRunner] Rigid visualization saved to {output_dir}")
        except Exception as e:
            print(f"[MCTSRunner] Warning: Failed to visualize rigid: {e}")
            import traceback
            traceback.print_exc()
    
    def _visualize_flex_results(self, candidates: List[BeamCandidate], summaries: List[Dict], output_dir: str):
        """Visualize flex stage beam candidates."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            save_flex_beam_results(candidates, summaries, output_dir)
            print(f"\n[MCTSRunner] Flex visualization saved to {output_dir}")
        except Exception as e:
            print(f"[MCTSRunner] Warning: Failed to visualize flex: {e}")
            import traceback
            traceback.print_exc()

    def _visualize_side_results(self, candidates: List[BeamCandidate], summaries: List[Dict], output_dir: str):
        """Visualize side stage beam candidates."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            save_side_beam_results(candidates, summaries, output_dir)
            print(f"\n[MCTSRunner] Side visualization saved to {output_dir}")
        except Exception as e:
            print(f"[MCTSRunner] Warning: Failed to visualize side: {e}")
            import traceback
            traceback.print_exc()

    def _load_nmr_predictor(self):
        """Helper to load the NMR predictor model safely."""
        if not hasattr(self, '_nmr_predictor') and self.config.nmr_model_ckpt:
            try:
                self._nmr_predictor = NMRPredictor(self.config.nmr_model_ckpt)
                print(f"  [NMR] Predictor loaded: {self.config.nmr_model_ckpt}")
            except Exception as e:
                print(f"  [NMR] Failed to load predictor: {e}")
                self._nmr_predictor = None
        return getattr(self, '_nmr_predictor', None)

    def _load_target_spectrum(self):
        """Helper to load the target spectrum safely."""
        if not hasattr(self, '_target_spectrum') and self.config.target_spectrum is not None:
            try:
                if isinstance(self.config.target_spectrum, str):
                    self._target_spectrum = load_target_spectrum_from_csv(self.config.target_spectrum)
                    print(f"  [NMR] Target spectrum loaded: {self.config.target_spectrum}")
                elif hasattr(self.config.target_spectrum, 'shape'):
                    self._target_spectrum = self.config.target_spectrum
            except Exception as e:
                print(f"  [NMR] Failed to load target spectrum: {e}")
                self._target_spectrum = None
        return getattr(self, '_target_spectrum', None)

    def _evaluate_nmr_state(self, state, nmr_predictor, target_spectrum):
        """Evaluate NMR cosine similarity for a given state. Returns (nmr_score, cos_sim) or (0, 0)."""
        try:
            raw_mol = mcts_state_to_raw_mol(state)
            if not raw_mol:
                return 0.0, 0.0
            pred_spectrum, _, _ = nmr_predictor.predict_spectrum(raw_mol)
            cos_sim = torch.nn.functional.cosine_similarity(
                pred_spectrum.unsqueeze(0),
                target_spectrum.unsqueeze(0),
            ).item()
            return max(0.0, cos_sim), cos_sim
        except Exception:
            return 0.0, 0.0

    def _make_nmr_score_fn(self, stage_name: str):
        nmr_predictor = self._load_nmr_predictor()
        target_spectrum = self._load_target_spectrum()
        if nmr_predictor is None or target_spectrum is None:
            return None
        if not hasattr(self, '_stage_nmr_cache'):
            self._stage_nmr_cache = {}

        def _score(stage_obj) -> float:
            try:
                state = stage_obj.state
                cache_key = (str(stage_name), state.state_signature())
            except Exception:
                return 0.0
            if cache_key in self._stage_nmr_cache:
                return float(self._stage_nmr_cache[cache_key])
            score, _cos = self._evaluate_nmr_state(state, nmr_predictor, target_spectrum)
            self._stage_nmr_cache[cache_key] = float(score)
            return float(score)

        return _score

    @staticmethod
    def _stage_result_key(stage_name: str) -> str:
        return {
            'flex': 'flex_result',
            'branch': 'branch_result',
            'side': 'side_result',
            'subst': 'subst_result',
        }.get(stage_name, f'{stage_name}_result')

    def _stage_iterations(self, stage_name: str) -> int:
        if stage_name == 'flex':
            return max(1, int(self.config.flex_iterations))
        if stage_name == 'branch':
            return max(1, int(getattr(self.config, 'branch_iterations', 30)))
        if stage_name == 'side':
            return max(1, int(getattr(self.config, 'side_iterations', 30)))
        if stage_name == 'subst':
            return max(1, int(getattr(self.config, 'subst_iterations', 12)))
        return max(1, int(self.config.iterations))

    def _stage_candidate_k(self, stage_name: str) -> int:
        if stage_name == 'branch':
            return max(1, int(getattr(self.config, 'branch_candidate_k', 8)))
        if stage_name == 'side':
            return max(1, int(getattr(self.config, 'side_candidate_k', 8)))
        if stage_name == 'subst':
            return max(1, int(getattr(self.config, 'subst_candidate_k', 6)))
        return max(1, int(getattr(self.config, 'max_candidates', 10)))

    def _stage_beam_width(self, stage_name: str) -> int:
        if stage_name == 'flex':
            return max(1, int(self.config.flex_beam_width))
        if stage_name in ('branch', 'side', 'subst'):
            return max(1, int(self.config.side_beam_width))
        return max(1, int(self.config.beam_width))

    def _stage_max_steps(self, stage_name: str, stage_obj: Any) -> int:
        if stage_name == 'flex':
            configured = int(getattr(self.config, 'flex_max_steps', 0) or 0)
            if configured > 0:
                return configured
            bridges_total = int(getattr(stage_obj, '_n_bridges_total', 0))
            return max(10, bridges_total * 3 + 6)
        if stage_name == 'branch':
            total = int(getattr(stage_obj, '_n_total', 0))
            return max(6, total * 3 + 6)
        if stage_name == 'side':
            total = int(getattr(stage_obj, '_n_sides_total', 0))
            return max(6, total * 3 + 6)
        if stage_name == 'subst':
            return 1
        return 80

    def _make_stage_evaluator(self, stage_name: str):
        nmr_fn = self._make_nmr_score_fn(stage_name)
        if stage_name == 'flex':
            return FlexStageEvaluator(nmr_score_fn=nmr_fn, nmr_weight=float(self.config.nmr_weight))
        if stage_name == 'branch':
            return BranchStageEvaluator(nmr_score_fn=nmr_fn, nmr_weight=float(self.config.nmr_weight))
        if stage_name == 'side':
            return SideStageEvaluator(nmr_score_fn=nmr_fn, nmr_weight=float(self.config.nmr_weight))
        if stage_name == 'subst':
            return SubstitutionStageEvaluator(nmr_score_fn=nmr_fn, nmr_weight=float(self.config.nmr_weight))
        return RigidStageEvaluator(max_cluster_size=int(self.config.max_cluster_size))

    def _prune_stage_candidates(self,
                                candidates: List[_StageBeamCandidate],
                                beam_width: int) -> List[_StageBeamCandidate]:
        if not candidates:
            return []
        unique: List[_StageBeamCandidate] = []
        seen = set()
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        for cand in ranked:
            try:
                sig = cand.stage.state.state_signature()
            except Exception:
                sig = id(cand)
            if sig in seen:
                continue
            seen.add(sig)
            unique.append(cand)
            if len(unique) >= max(1, int(beam_width)):
                break
        return unique

    def _build_stage_provider(self, stage_name: str, base_cand: BeamCandidate):
        state = base_cand.state.copy()
        state.stage = stage_name
        state.stage_step = 0
        state.stage_mode = 'default'
        state.stage_meta = {}

        allocator = base_cand.info.get('allocator')
        allocation_result = base_cand.info.get('allocation_result')
        if stage_name == 'flex':
            if allocator is None:
                allocator = FlexAllocator(self.nodes_csv)
                allocator.allocate()
            alloc_result = copy.deepcopy(allocation_result) if allocation_result is not None else allocator._result
            return FlexStage(state, self.init_data['su_counts'], allocation_result=alloc_result), allocator
        if stage_name == 'branch':
            alloc_result = copy.deepcopy(allocation_result) if allocation_result is not None else (allocator._result if allocator is not None else None)
            if alloc_result is None:
                return None, None
            branch_specs = alloc_result.branch_chains if alloc_result else []
            return BranchStage(state, branch_specs), allocator
        if stage_name == 'side':
            alloc_result = copy.deepcopy(allocation_result) if allocation_result is not None else (allocator._result if allocator is not None else None)
            if alloc_result is None:
                return None, None
            side_allocator = copy.deepcopy(allocator) if allocator is not None else FlexAllocator(su_counts=self.init_data['su_counts'])
            side_allocator._result = alloc_result
            return SideStage(state, side_allocator), side_allocator
        if stage_name == 'subst':
            return SubstitutionStage(
                state,
                self.init_data['su_counts'],
                nodes_csv=self.nodes_csv,
                n_variants=max(1, int(getattr(self.config, 'subst_n_variants', 3))),
                seed_base=int(self.config.seed or 0),
                nmr_eval_fn=None,
            ), allocator
        return None, allocator

    def _run_stage_with_engine(self,
                               stage_name: str,
                               input_candidates: List[BeamCandidate]) -> Dict:
        if not input_candidates:
            return {'beam_candidates': [], 'best_candidate': None, 'summaries': []}

        evaluator = self._make_stage_evaluator(stage_name)
        engine = StageSearchEngine(
            iterations=self._stage_iterations(stage_name),
            c_puct=float(self.config.c_puct),
            candidate_k=self._stage_candidate_k(stage_name),
            rollout_depth=max(1, int(self.config.max_depth // 4)),
        )
        beam_width = self._stage_beam_width(stage_name)
        action_branching = max(1, int(getattr(self.config, 'stage_action_branching', 2)))
        all_results: List[_StageBeamCandidate] = []
        result_key = self._stage_result_key(stage_name)

        for base_cand in input_candidates:
            provider, allocator = self._build_stage_provider(stage_name, base_cand)
            if provider is None:
                continue
            initial_info = copy.deepcopy(base_cand.info)
            if allocator is not None:
                initial_info['allocator'] = allocator
            if stage_name == 'flex':
                initial_info['allocation_result'] = copy.deepcopy(getattr(provider, 'allocation_result', None))
            initial_stage_score = float(evaluator.evaluate(provider))
            candidates = [_StageBeamCandidate(
                stage=provider,
                base_score=float(base_cand.score),
                stage_score=initial_stage_score,
                action_sequence=list(base_cand.action_sequence),
                info=initial_info,
                completed=bool(provider.is_done()),
            )]
            max_steps = self._stage_max_steps(stage_name, provider)

            for _ in range(max_steps):
                if all(c.completed or c.stage.is_done() for c in candidates):
                    break
                new_candidates: List[_StageBeamCandidate] = []
                for cand in candidates:
                    if cand.completed or cand.stage.is_done():
                        cand.completed = True
                        new_candidates.append(cand)
                        continue
                    actions = engine.search_actions(
                        cand.stage,
                        evaluator,
                        branch_width=min(action_branching, beam_width),
                        candidate_k=self._stage_candidate_k(stage_name),
                    )
                    if not actions:
                        cand.completed = True
                        new_candidates.append(cand)
                        continue
                    expanded = False
                    for action in actions:
                        next_stage = cand.stage.clone()
                        if not next_stage.step(copy.deepcopy(action)):
                            continue
                        next_info = copy.deepcopy(cand.info)
                        if allocator is not None:
                            next_info['allocator'] = allocator
                        next_info['stage'] = stage_name
                        stage_result = next_stage.get_result()
                        next_info[result_key] = stage_result
                        if stage_name == 'flex':
                            next_info['allocation_result'] = copy.deepcopy(getattr(next_stage, 'allocation_result', None))
                        if stage_name == 'subst':
                            next_info['subst_summary'] = dict(stage_result.get('applied', {}) or {})
                            next_info['subst_remaining'] = dict(stage_result.get('remaining', {}) or {})
                            next_info['subst_complete'] = bool(stage_result.get('complete', False))
                            next_info['subst_remaining_total'] = int(stage_result.get('remaining_total', 0))
                            next_info['subst_l1_delta'] = int(stage_result.get('l1_delta', 0))
                            next_info['subst_success'] = bool(stage_result.get('success', False))
                            next_info['seed'] = int(stage_result.get('seed', 0))
                        if getattr(evaluator, 'nmr_score_fn', None) is not None and stage_name in {'flex', 'branch', 'side', 'subst'}:
                            try:
                                next_info['nmr_score'] = float(evaluator.nmr_score_fn(next_stage))
                                next_info['nmr_cos_sim'] = float(next_info['nmr_score'])
                            except Exception:
                                next_info['nmr_score'] = 0.0
                                next_info['nmr_cos_sim'] = 0.0
                        new_candidates.append(_StageBeamCandidate(
                            stage=next_stage,
                            base_score=float(cand.base_score),
                            stage_score=float(evaluator.evaluate(next_stage)),
                            action_sequence=list(cand.action_sequence) + [copy.deepcopy(action)],
                            info=next_info,
                            completed=bool(next_stage.is_done()),
                        ))
                        expanded = True
                    if not expanded:
                        cand.completed = True
                        new_candidates.append(cand)
                candidates = self._prune_stage_candidates(new_candidates, beam_width)

            all_results.extend(candidates)

        final_beam: List[BeamCandidate] = []
        for cand in sorted(all_results, key=lambda c: c.score, reverse=True)[:beam_width]:
            info = copy.deepcopy(cand.info)
            stage_result = cand.stage.get_result()
            info[result_key] = stage_result
            if stage_name == 'flex':
                info['allocation_result'] = copy.deepcopy(getattr(cand.stage, 'allocation_result', None))
            if stage_name == 'subst':
                info['subst_summary'] = dict(stage_result.get('applied', {}) or {})
                info['subst_remaining'] = dict(stage_result.get('remaining', {}) or {})
                info['subst_complete'] = bool(stage_result.get('complete', False))
                info['subst_remaining_total'] = int(stage_result.get('remaining_total', 0))
                info['subst_l1_delta'] = int(stage_result.get('l1_delta', 0))
                info['subst_success'] = bool(stage_result.get('success', False))
                info['seed'] = int(stage_result.get('seed', 0))
            if getattr(evaluator, 'nmr_score_fn', None) is not None and stage_name in {'flex', 'branch', 'side', 'subst'} and 'nmr_score' not in info:
                try:
                    info['nmr_score'] = float(evaluator.nmr_score_fn(cand.stage))
                    info['nmr_cos_sim'] = float(info['nmr_score'])
                except Exception:
                    info['nmr_score'] = 0.0
                    info['nmr_cos_sim'] = 0.0
            final_beam.append(BeamCandidate(
                state=cand.stage.state.copy(),
                score=float(cand.score),
                action_sequence=list(cand.action_sequence),
                info=info,
            ))

        summaries = [c.info.get(result_key, {}) for c in final_beam]
        return {
            'beam_candidates': final_beam,
            'best_candidate': final_beam[0] if final_beam else None,
            'summaries': summaries,
        }

    def run_all_stages(self, output_dir: Optional[str] = None) -> Dict:
        """
        Run the complete pipeline: rigid -> flex -> branch -> side -> substitution.
        Handles intermediate state passing and returns the final side stage result.
        
        Args:
            output_dir: Base directory for outputs. Subdirectories (rigid, flex, branch, side, subst) 
                       will be created automatically.
        """
        import os
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            rigid_dir = os.path.join(output_dir, 'rigid')
            flex_dir = os.path.join(output_dir, 'flex')
            branch_dir = os.path.join(output_dir, 'branch')
            side_dir = os.path.join(output_dir, 'side')
            subst_dir = os.path.join(output_dir, 'subst')
        else:
            rigid_dir = flex_dir = branch_dir = side_dir = subst_dir = None
            
        print(f"\n{'='*60}")
        print("MCTS Pipeline: rigid -> flex -> branch -> side -> substitution")
        print(f"{'='*60}")
        
        # 1. Rigid Stage
        rigid_result = self.run_rigid_stage(output_dir=rigid_dir)
        if not rigid_result.get('beam_candidates'):
            print("[Pipeline] FAILED at rigid stage")
            return {'status': 'failed', 'stage': 'rigid'}
            
        # 2. Flex Stage
        flex_result = self.run_flex_stage(rigid_result, output_dir=flex_dir)
        if not flex_result.get('beam_candidates'):
            print("[Pipeline] FAILED at flex stage")
            return {'status': 'failed', 'stage': 'flex'}
            
        # 3. Branch Stage
        branch_result = self.run_branch_stage(flex_result, output_dir=branch_dir)
        if not branch_result.get('beam_candidates'):
            print("[Pipeline] FAILED at branch stage")
            return {'status': 'failed', 'stage': 'branch'}
            
        # 4. Side Stage
        side_result = self.run_side_stage(branch_result, output_dir=side_dir)
        if not side_result.get('beam_candidates'):
            print("[Pipeline] FAILED at side stage")
            return {'status': 'failed', 'stage': 'side'}

        # 5. Substitution Stage (Carbonyls)
        subst_result = self.run_substitution_stage(side_result, output_dir=subst_dir)
   
        self._print_final_summary(subst_result)
        
        return subst_result
        
    def run_substitution_stage(self, side_result: Dict, output_dir: Optional[str] = None) -> Dict:
        """
        Run substitution stage using the unified stage-search engine.
        """
        print(f"\n[SubstStage] Starting (beam={self.config.side_beam_width}, iter={self._stage_iterations('subst')})")

        side_candidates = side_result.get('beam_candidates', [])
        if not side_candidates:
            return {'beam_candidates': [], 'best_candidate': None,
                    'summaries': []}
        result = self._run_stage_with_engine('subst', side_candidates)
        subst_candidates = list(result.get('beam_candidates', []))
        if not subst_candidates:
            return result

        subst_candidates.sort(key=self._subst_result_rank_key, reverse=True)
        subst_candidates = subst_candidates[:self.config.side_beam_width]

        print(f"[SubstStage] Complete: {len(subst_candidates)} candidates")
        for i, c in enumerate(subst_candidates[:3]):
            nmr_str = f", NMR={c.info.get('nmr_score', 0.0):.3f}"
            print(f"  #{i+1}: score={c.score:.2f}{nmr_str}")
        summaries_sorted = [c.info for c in subst_candidates]
        result = {
            'beam_candidates': subst_candidates,
            'best_candidate': (subst_candidates[0] if subst_candidates else None),
            'summaries': summaries_sorted,
        }

        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            try:
                from .visualization import save_subst_beam_results
                save_subst_beam_results(
                    subst_candidates, summaries_sorted, output_dir)
            except Exception as e:
                print(f"[MCTSRunner] Warning: Failed to visualize subst: {e}")

        return result

    def _print_final_summary(self, final_result: Dict):
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE - Final Summary")
        print(f"{'='*60}")
        
        best = final_result.get('best_candidate')
        if not best:
            print("  No valid candidates produced.")
            return
            
        state = best.state
        
        n_clusters = len(state.graph.clusters)
        n_placed = sum(1 for c in state.graph.clusters if c.placed)
        n_components = len(set(state._find(c.id) for c in state.graph.clusters if c.placed))
        
        # Edge counts
        n_rigid = len(state.graph.rigid)
        n_flex = len(state.graph.flex)
        n_branch = len(state.graph.branch)
        n_side = len(state.graph.side)
        
        # Chain node counts
        n_chain_nodes = len(state.graph.chains)
        
        print(f"\n[Structure]")
        print(f"  Clusters: {n_placed}/{n_clusters} placed, {n_components} component(s)")
        print(f"  Edges: rigid={n_rigid}, flex={n_flex}, branch_total={n_branch}, side={n_side}")
        print(f"  Chain nodes: {n_chain_nodes}")
        overlaps = find_overlapping_axials(state.graph)
        if overlaps:
            print(f"  Warning: overlapping axial positions detected: {len(overlaps)}")
        
        # SU distribution
        su_dist = state.get_su_distribution()
        print(f"\n[SU Distribution]")
        su_groups = {
            'Aromatic (10-13)': [10, 11, 12, 13],
            'Aliphatic (22-25)': [22, 23, 24, 25],
        }
        for group_name, su_list in su_groups.items():
            counts = [f"{su}:{su_dist.get(su, 0)}" for su in su_list if su_dist.get(su, 0) > 0]
            if counts:
                print(f"  {group_name}: {', '.join(counts)}")

        target_su = dict(self.init_data.get('su_counts', {}))
        all_su = sorted(set(target_su) | set(su_dist))
        actual_parts = [f"{su}:{su_dist.get(su, 0)}" for su in all_su if su_dist.get(su, 0) > 0]
        target_parts = [f"{su}:{target_su.get(su, 0)}" for su in all_su if target_su.get(su, 0) > 0]
        delta_parts = [f"{su}:{delta:+d}" for su, delta in compute_su_delta(su_dist, target_su).items()]
        print(f"\n[SU Comparison]")
        print(f"  Actual: {', '.join(actual_parts) if actual_parts else 'none'}")
        print(f"  Target: {', '.join(target_parts) if target_parts else 'none'}")
        print(f"  Delta (actual-target): {', '.join(delta_parts) if delta_parts else 'none'}")
        
        # Score breakdown
        print(f"\n[Scoring]")
        print(f"  Final score: {best.score:.2f}")
        
        # NMR info if available
        nmr_score = best.info.get('nmr_score', 0)
        if nmr_score > 0:
            print(f"  NMR cosine similarity: {nmr_score:.4f}")
        
        # Stage results
        fr = best.info.get('flex_result', {})
        br = best.info.get('branch_result', {})
        sr = best.info.get('side_result', {})
        
        print(f"\n[Stage Results]")
        if fr:
            print(f"  Flex: {fr.get('connections_made', 0)}/{fr.get('bridges_total', 0)} bridges, "
                  f"connected={fr.get('all_connected', False)}")
        if br:
            print(f"  Branch: {br.get('branches_placed', 0)}/{br.get('branches_total', 0)} rings")
        if sr:
            print(f"  Side: {sr.get('sides_placed', 0)}/{sr.get('sides_total', 0)} chains")

        subst_summary = best.info.get('subst_summary', {})
        subst_remaining = best.info.get('subst_remaining', {})
        subst_complete = bool(best.info.get('subst_complete', False))
        subst_l1_delta = int(best.info.get('subst_l1_delta', 0))
        if subst_summary or subst_remaining or 'subst_complete' in best.info:
            print(f"\n[Substitution]")
            if subst_summary:
                parts = [f"{su}:{cnt:+d}" for su, cnt in sorted(subst_summary.items())]
                print(f"  Delta (after-before): {', '.join(parts)}")
            else:
                print("  Delta (after-before): none")
            if subst_remaining:
                parts = [f"{su}:{cnt:+d}" for su, cnt in sorted(subst_remaining.items())]
                print(f"  Final delta vs target: {', '.join(parts)}")
            else:
                print("  Final delta vs target: none")
            print(f"  Match target: {subst_complete}, L1 delta={subst_l1_delta}")
        
        print(f"{'='*60}")

    def run_branch_stage(self, flex_result: Dict, output_dir: Optional[str] = None) -> Dict:
        """
        Run branch stage using the unified stage-search engine.
        Branch resource forms still come directly from FlexAllocator; search
        only chooses valid placement positions for those pre-allocated rings.
        """
        print(f"\n[BranchStage] Starting (beam={self.config.side_beam_width}, iter={self._stage_iterations('branch')})")

        flex_candidates = flex_result.get('beam_candidates', [])
        if not flex_candidates:
            print("[BranchStage] No flex candidates to build on!")
            return {'beam_candidates': [], 'best_candidate': None, 'summaries': []}
        result = self._run_stage_with_engine('branch', flex_candidates)
        branch_candidates = list(result.get('beam_candidates', []))
        if not branch_candidates:
            print("[BranchStage] No candidates produced!")
            return result

        print(f"[BranchStage] Complete: {len(branch_candidates)} candidates")
        for i, c in enumerate(branch_candidates[:3]):
            br = c.info.get('branch_result', {})
            nmr = c.info.get('nmr_score', 0.0)
            print(f"  #{i+1}: {br.get('branches_placed',0)}/{br.get('branches_total',0)} rings, "
                  f"NMR={nmr:.3f}, score={c.score:.2f}")

        if output_dir:
            self._visualize_branch_results(branch_candidates, result.get('summaries', []), output_dir)

        return result

    def _visualize_branch_results(self, candidates: List[BeamCandidate], summaries: List[Dict], output_dir: str):
        """Visualize branch stage beam candidates."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        try:
            save_branch_beam_results(candidates, summaries, output_dir)
            print(f"\n[MCTSRunner] Branch visualization saved to {output_dir}")
        except Exception as e:
            print(f"[MCTSRunner] Warning: Failed to visualize branch: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point — runs rigid + flex + branch + side stages"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MCTS Rigid + Flex + Branch + Side Stage Search')
    parser.add_argument('--nodes', type=str, required=True, help='Path to nodes CSV')
    parser.add_argument('--spectrum', type=str, default=None, help='Path to spectrum CSV')
    parser.add_argument('--iterations', type=int, default=100, help='MCTS iterations (rigid)')
    parser.add_argument('--flex_iterations', type=int, default=60, help='MCTS iterations (flex)')
    parser.add_argument('--branch_iterations', type=int, default=30, help='MCTS iterations (branch)')
    parser.add_argument('--side_iterations', type=int, default=30, help='MCTS iterations (side)')
    parser.add_argument('--subst_iterations', type=int, default=12, help='MCTS iterations (subst)')
    parser.add_argument('--subst_n_variants', type=int, default=3, help='Number of substitution trial variants per candidate')
    parser.add_argument('--side_beam_width', type=int, default=5, help='Side/Branch stage beam width')
    parser.add_argument('--nmr_model', type=str, default=None, help='G2S NMR model checkpoint')
    parser.add_argument('--target_spectrum', type=str, default=None, help='Target NMR spectrum CSV')
    parser.add_argument('--nmr_weight', type=float, default=20.0, help='NMR score weight')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--output', type=str, default='output/', help='Output directory')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['rigid', 'flex', 'branch', 'side', 'subst', 'both', 'all'],
                        help='Which stage(s) to run (both=rigid+flex, all=rigid+flex+branch+side+subst)')
    
    args = parser.parse_args()
    
    config = MCTSConfig(
        iterations=args.iterations,
        flex_iterations=args.flex_iterations,
        branch_iterations=args.branch_iterations,
        side_iterations=args.side_iterations,
        subst_iterations=args.subst_iterations,
        subst_n_variants=args.subst_n_variants,
        side_beam_width=args.side_beam_width,
        nmr_model_ckpt=args.nmr_model,
        target_spectrum=args.target_spectrum,
        nmr_weight=args.nmr_weight,
        seed=args.seed,
    )
    
    runner = MCTSRunner(args.nodes, args.spectrum, config)
    
    rigid_dir = os.path.join(args.output, 'rigid')
    flex_dir = os.path.join(args.output, 'flex')
    branch_dir = os.path.join(args.output, 'branch')
    side_dir = os.path.join(args.output, 'side')
    subst_dir = os.path.join(args.output, 'subst')
    
    rigid_result = None
    flex_result = None
    branch_result = None
    side_result = None
    subst_result = None
    
    # Pipeline: rigid → flex → branch → side → subst
    if args.stage in ('rigid', 'flex', 'branch', 'side', 'subst', 'both', 'all'):
        rigid_result = runner.run_rigid_stage(output_dir=rigid_dir)
    
    if args.stage in ('flex', 'branch', 'side', 'subst', 'both', 'all') and rigid_result:
        flex_result = runner.run_flex_stage(rigid_result, output_dir=flex_dir)
    
    if args.stage in ('branch', 'side', 'subst', 'all') and flex_result:
        branch_result = runner.run_branch_stage(flex_result, output_dir=branch_dir)
    
    if args.stage in ('side', 'subst', 'all') and branch_result:
        side_result = runner.run_side_stage(branch_result, output_dir=side_dir)

    if args.stage in ('subst', 'all') and side_result:
        subst_result = runner.run_substitution_stage(side_result, output_dir=subst_dir)
    else:
        subst_result = None
    
    print(f"\nResults saved to {args.output}")
    return {'rigid': rigid_result, 'flex': flex_result, 'branch': branch_result, 'side': side_result, 'subst': subst_result}

if __name__ == '__main__':
    main()
