# RL_MTCS

Stages: rigid -> flex -> ring -> side -> SU26/SU30 substitution.
Legal partial molecules can be exported. Completion always uses the original
task inventory, including units outside the final anchored main component.

## Responsibilities

| Module | Responsibility |
| --- | --- |
| `config.py` | Current configuration; no obsolete budget arguments |
| `mdp/types.py`, `mdp/state.py`, `mdp/sites.py` | Tasks, state and read-only queries |
| `mdp/components.py`, `mdp/signatures.py` | Anchored scope and policy-aware equivalence |
| `actions/` | Independent rigid, flex, ring, side and substitution actions |
| `mdp/transition.py` | Atomic mutation and validated action application |
| `geometry.py`, `validation.py` | Honeycomb geometry and molecular legality |
| `rewards/demand.py`, `rewards/scoring.py` | Frozen topology demand and five reward terms |
| `policy/mcts.py`, `policy/checkpoints.py` | PUCT selection and local rollback |
| `policy/rollout.py` | One greedy continuation of every remaining stage |
| `storage/state_delta.py` | Compact encoding and exact replay, not molecular policy |
| `io/`, `reporting/` | CSV tasks, exports, resource and count-only diagnostics |

`legal_actions(state, phase, limit)` returns complete validated `ConnectionAction`
objects. `apply_action(state, action)` returns a new state and rejects stale
actions. Actions contain all endpoint bindings, selected conformers, poses and
the exact atomic effect; they retain no live state snapshots.

Rigid and flex track separate previous axes. Vertical means `(0, +/-1)`;
the other four directions are horizontal. Prefer the opposite successful axis;
use the other axis only if the preferred one has no legal action. Flex component
merges take precedence over internal placement and direction preference.

The post-flex main-component convex hull remains fixed during ring and side
attachment. Rings try outside, crossing and inside placements, in that order
across all remaining ring units. Hull regions are preferences, not legality
constraints; boundary nodes can be used if the full connection passes its audit.
Complex sides likewise prefer exterior placements before spatial fallback;
simple sides prefer inside. The legal candidate cap applies across all regions.

Generated side chains are embedded from their actual attachment entry. Each
partial branch checks current occupancy, honeycomb directions and explicit bond
intersections before extending; constrained branches are resolved first. Node GIDs,
labels and prescribed edges stay fixed. Explicit input coordinates still allow
only whole-unit rotations and reflections.

Side candidates offer one legal shape per entry and one outcome per topology
before returning for additional shapes. Equivalent
occupied geometry does not consume another slot. This keeps internal bends in
the ten-candidate set without scanning or retaining all possible conformers.
All failed locations remain searchable; quick forward still commits the first
legal outcome without scoring or comparing future side placements.

When direct side attachment is exhausted, formal side search can reshape one
already placed side while preserving its attachment node, port and contract.
The reshape and one new attachment form a single audited action; a reshape
alone is never a selectable search action. Quick forward uses direct actions.

## Search And Forward Evaluation

Each connection admits at most `connection_candidate_limit` legal actions.
A selection round compares at most that many distinct results. Invalid scans
do not consume candidate slots; no global time, node or iteration budgets exist.
Every admitted root action receives a continuation evaluation before the round
selects a result. Deeper branches use PUCT. Logs distinguish admitted root
actions, evaluated root actions and retained representative results.
Formal candidate expansion orders admitted actions by the existing reward;
this scoring never invokes forward rollout. Quick continuation does not rank.
Search statistics and compact pending effects live only for the current round.

Only the selected rigid result and selected flex checkpoints run full forward
evaluation. Flex checkpoints occur after four successful unit connections, and
at a short or zero-progress stage end. Forward uses the first legal action at
each step, with no internal MCTS, NMR or reward ranking, then audits the result.
Rejected whole results trigger checkpoint-local retries with reproducible
different candidate ordering. Individual connections can be reused in a new
combination. Exhausted retries choose the best audited evaluated result.

Defaults: candidate limit 10, checkpoint interval 4, minimum forward completion
0.80, maximum rollbacks 4, SU13 axial clearance radius 3. The five base reward
weights remain `(0.40, 0.15, 0.20, 0.20, 0.05)`.

Space guidance adapts to the input rather than a sample name. At the start of
each rigid round, flex checkpoint or ring/side stage, count the remaining
topology nodes and required aromatic ports. Divide these by the original
aromatic-plus-flex node count and original staging SU13 capacity. The larger
ratio, clipped to `[0, 1]`, is the space pressure `g`. The clearance weight is
`base_clearance * g`; its released weight moves to shape. Completion, NMR and
forward base weights stay fixed. An unavailable NMR term is then normalized
out as before. SU26/SU30 requests require no clearance demand.

All candidate evaluations and retries from the same checkpoint share this
frozen demand. Failed scans, occupied entrances and excluded units cannot
reduce its reference capacity or silently erase unfinished tasks. Only later
stage/checkpoint boundaries update demand; ended phases retain their original
unfinished-task reports but stop reserving space for actions no longer run.
Within a long ring/side round, the weights intentionally remain fixed until
stage end. The exported `remaining_spatial_demand` reports the next boundary;
the reward's `spatial_demand` identifies the context actually used to select it.

Shape is `0.8 * compactness + 0.2 * aspect`, with compactness
`1 / (1 + Rg / sqrt(N))` in the Cartesian projection and aspect the inverse
principal-axis ratio. Rigid components are measured separately; later stages
use the anchored main component, excluding staging distances. Final midpoint
positions participate in shape. There are no per-candidate hull or void scans.
Exact reward ties prefer usable entrances only up to remaining port demand;
completed attachment tasks have no entrance tie advantage. Quick forward
remains unscored. Rewards export base/effective weights, demand counts,
weighted contributions and compactness/aspect observations.

## Identity And Geometry

Logical GIDs remain tied to original tasks. Aromatic GID rebinding changes
physical slots and port owners/directions, never the logical contract endpoints.
`source_problem` retains original identities; `slot_id` identifies the physical
slot. SU13 is a pending aromatic label and becomes SU5--SU11 only after actual
external edges are committed. NMR consumes the committed nodes, edges and labels.

All ordinary nodes, including hetero connectors, occupy honeycomb vertices;
all ordinary internal and external edges follow honeycomb directions. There
is one explicit exception: each four-valent SU25 registers exactly one fixed
terminal SU22 branch at an adjacent hexagon centre. The registration is bound to
logical GIDs before search; other branches, including another SU22, remain ordinary
honeycomb vertices. Collisions and edge intersections are still prohibited.
Adjacent vertices without a prescribed edge are allowed. Only explicit
topology/contract edges contribute to connectivity, degree and label activation.
Five-member SU26/SU30 midpoint geometry is audited only for
its explicitly recorded affected edges and aromatic unit.

`legal` combines independent identity, label, topology and geometry checks.
`all_complete` reports task completion independently. `input_valid` and
`input_errors` expose geometrically unsupported input and missing boundary tasks;
these resources retain their original inventory and denominators. Chemical port
semantics belong to the inverse pipeline. Assembly preserves the supplied SU
identities and explicit topology, including `5-2-2-5` and `9-3-3-9`, without
rechecking chemical port combinations.

## Run

```python
from RL_MTCS import AssemblyConfig, assemble_from_csv

result = assemble_from_csv(
    nodes_csv=nodes_csv,
    topology_specs_csv=topology_csv,
    output_dir=str(output),
    config=AssemblyConfig(save_stage_images=False),
)
```

```bash
python -m RL_MTCS.run_ym3 --preflight-only
python -m RL_MTCS.run_ym3
python -m pytest tests/test_rl_mtcs*.py tests/test_aromatic_coordination.py tests/test_flex_reconciliation.py
```

There are no forwarding modules for removed internal paths. Restart notebook
kernels after updating this package. Search returns the best result among the
searched candidates, not a proof of a global maximum.
