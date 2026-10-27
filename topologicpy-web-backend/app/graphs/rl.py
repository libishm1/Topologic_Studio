"""Tabular Q-learning over the navigation graph.

Started as Classic's algorithm with two corrections (a clean greedy rollout,
and ties broken under the caller's seed rather than by adjacency order). The
2026-10 review found it could not learn a route on a real model at all, for
reasons that were each small and together fatal:

* the caller seeded the fire at the evacuee's own start node, so a flood front
  moving one hop per step chased the agent at exactly its own speed and any
  exploratory move ended the episode (see ``routers/simulation.py``);
* ``gamma = 0.9`` discounts the +10 exit reward to 0.004 at 75 hops, which is
  a short walk on the Duplex - the agent could not see the exit;
* terminal transitions bootstrapped from the next node's Q row, so a node that
  burns late was valued by what happened there earlier;
* walls were ignored, so the policy walked through edges A* refuses;
* the reported "reward" was a sum of Q-values, not a return.

The fixes keep the method tabular and explainable. Step cost is proportional
to edge length, so the agent optimises the same metres A* does. Learning is
accelerated by potential-based shaping on the walls-respecting distance to the
exit (Ng, Harada & Russell 1999), which changes how fast the policy is found,
not which policy is optimal: the exit has potential 0 and every other node is
negative, so shaping never makes burning preferable to escaping.

With a fire the state is ``(node, step)``. Classic used the node alone, but
the fire makes the reward time-dependent: arriving at a corridor one step late
burns, one step early does not. A node-only table pins that -10 on the
corridor itself, and on a tight race the agent learned to avoid the only route
that escapes. Without a fire the problem is stationary and the state stays the
node, so experience generalises across time.

Untried actions start at their one-step shaped value instead of 0, so the
first greedy moves already head for the exit. Tabular Q-learning converges from
any initialisation; this only changes how much exploration it takes.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..store import NavGraph

EXIT_REWARD = 10.0
BURN_REWARD = -10.0
#: Cost of one median-length edge, as in Classic. Longer edges cost more.
STEP_COST = 0.1
#: Training curve resolution returned to the client.
CURVE_POINTS = 50


@dataclass
class RLResult:
    path: List[int]
    reached_exit: bool
    episodes: int
    #: Undiscounted environment return of the greedy rollout (no shaping).
    reward: float
    #: The greedy rollout stepped onto a node that was already burning.
    burned: bool = False
    #: Length of the greedy rollout in model units.
    length: float = 0.0
    #: Walls-respecting shortest distance start -> exit, ignoring fire.
    optimal_length: Optional[float] = None
    #: Share of the final fifth of training episodes that reached the exit.
    success_rate: float = 0.0
    #: Training success rate, bucketed to at most ``CURVE_POINTS`` values.
    curve: List[float] = field(default_factory=list)
    #: Fewest moves that escape the fire, or None when no route can. Lets the
    #: UI tell "the agent failed to learn" from "there was nothing to learn".
    escape_steps: Optional[int] = None
    note: Optional[str] = None


def exit_distances(
    graph: NavGraph, exit_node: int, blocked: Optional[np.ndarray] = None
) -> np.ndarray:
    """Shortest distance from every node to ``exit_node`` over open edges."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    n = graph.node_count
    if graph.edge_count == 0:
        out = np.full(n, np.inf)
        out[exit_node] = 0.0
        return out
    keep = np.ones(graph.edge_count, dtype=bool) if blocked is None else ~blocked
    edges = graph.edges[keep]
    # A zero-length edge would vanish from a sparse matrix; nudge it.
    lengths = np.maximum(graph.edge_lengths[keep].astype(np.float64), 1e-9)
    matrix = csr_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(n, n))
    return dijkstra(matrix, directed=False, indices=exit_node)


def earliest_escape(
    graph: NavGraph,
    start: int,
    exit_node: int,
    ignite_time: Optional[Dict[int, int]],
    max_steps: int,
    blocked: Optional[np.ndarray] = None,
) -> Optional[int]:
    """Fewest moves from ``start`` to ``exit_node`` that never enter a burning
    node, under the same rule the agent is rewarded by (move ``t`` into a node
    that ignited at or before ``t`` burns). None when no such route exists.

    Plain BFS is exact here: the fire only spreads, so reaching a node earlier
    is never worse than reaching it later.
    """
    if start == exit_node:
        return 0
    never = 1 << 30
    ignite = ignite_time or {}
    indptr, neighbours, edge_ids = graph.csr()
    bounds, target, eids = indptr.tolist(), neighbours.tolist(), edge_ids.tolist()
    closed = None if blocked is None else blocked.tolist()
    seen = {start}
    frontier = [start]
    for t in range(max_steps):
        nxt: List[int] = []
        for node in frontier:
            for k in range(bounds[node], bounds[node + 1]):
                other = target[k]
                if other in seen or (closed is not None and closed[eids[k]]):
                    continue
                if t >= ignite.get(other, never):
                    continue
                if other == exit_node:
                    return t + 1
                seen.add(other)
                nxt.append(other)
        if not nxt:
            return None
        frontier = nxt
    return None


def q_learning_path(
    graph: NavGraph,
    start: int,
    exit_node: int,
    ignite_time: Optional[Dict[int, int]] = None,
    episodes: int = 200,
    max_steps: int = 200,
    epsilon: float = 0.2,
    alpha: float = 0.5,
    gamma: float = 0.99,
    seed: Optional[int] = None,
    blocked: Optional[np.ndarray] = None,
    shaping: bool = True,
) -> RLResult:
    n = graph.node_count
    if n == 0 or not (0 <= start < n) or not (0 <= exit_node < n):
        return RLResult([], False, 0, 0.0)
    if start == exit_node:
        return RLResult([start], True, 0, 0.0, optimal_length=0.0)

    distance = exit_distances(graph, exit_node, blocked)
    if not math.isfinite(distance[start]):
        return RLResult(
            [start], False, 0, 0.0,
            note="The exit cannot be reached from the start without crossing a wall.",
        )

    rng = random.Random(seed)
    unit = graph.step_size()

    # Plain Python lists: the loop below indexes them hundreds of thousands of
    # times, and numpy scalar indexing is several times slower than a list.
    indptr, neighbours, edge_ids = graph.csr()
    target = neighbours.tolist()
    cost = (graph.edge_lengths[edge_ids].astype(np.float64) / unit * STEP_COST).tolist()
    open_slot = None if blocked is None else (~blocked[edge_ids]).tolist()
    bounds = indptr.tolist()

    def slots_of(node: int) -> List[int]:
        lo, hi = bounds[node], bounds[node + 1]
        if open_slot is None:
            return list(range(lo, hi))
        return [k for k in range(lo, hi) if open_slot[k]]

    slots = [slots_of(i) for i in range(n)]
    # Potential: minus the remaining distance, in the same units as the step
    # cost. Nodes cut off from the exit get the worst finite potential.
    finite = np.isfinite(distance)
    worst = float(distance[finite].max()) if finite.any() else 0.0
    potential = (
        (-np.where(finite, distance, worst) / unit * STEP_COST).tolist()
        if shaping
        else [0.0] * n
    )

    never = 1 << 30
    ignite = ignite_time or {}
    timed = bool(ignite)
    horizon = max_steps + 1
    source = np.repeat(np.arange(n), np.diff(indptr)).tolist()
    prior = [
        -cost[k] + gamma * potential[target[k]] - potential[source[k]]
        for k in range(len(target))
    ]
    # Sparse table keyed by slot (and step, when the fire makes time matter).
    q: Dict[int, float] = {}

    def key(k: int, t: int) -> int:
        return k * horizon + t if timed else k

    def value(k: int, t: int) -> float:
        return q.get(key(k, t), prior[k])

    def greedy(node: int, t: int, avoid: Optional[set] = None) -> Optional[int]:
        options = slots[node]
        if avoid:
            options = [k for k in options if target[k] not in avoid]
        if not options:
            return None
        top = -math.inf
        winners: List[int] = []
        for k in options:
            v = value(k, t)
            if v > top:
                top = v
                winners = [k]
            elif v == top:
                winners.append(k)
        return rng.choice(winners)

    successes: List[int] = []
    for _ in range(max(1, episodes)):
        state = start
        reached = 0
        for t in range(max_steps):
            options = slots[state]
            if not options:
                break
            if rng.random() < epsilon:
                k = options[rng.randrange(len(options))]
            else:
                k = greedy(state, t)
            nxt = target[k]

            reward = -cost[k]
            done = False
            if t >= ignite.get(nxt, never):
                reward = BURN_REWARD
                done = True
            elif nxt == exit_node:
                reward = EXIT_REWARD
                done = True
                reached = 1

            # Terminal states have zero potential, and the last allowed step is
            # terminal (literature review, finding W5): Ng, Harada & Russell's
            # guarantee needs Phi = 0 at terminals, and bootstrapping past the
            # horizon read untrained prior values.
            shaped = reward + gamma * (0.0 if done else potential[nxt]) - potential[state]
            if done or t == max_steps - 1:
                td_target = shaped
            else:
                nxt_options = slots[nxt]
                best_next = max(value(j, t + 1) for j in nxt_options) if nxt_options else 0.0
                td_target = shaped + gamma * best_next
            current = value(k, t)
            q[key(k, t)] = current + alpha * (td_target - current)

            state = nxt
            if done:
                break
        successes.append(reached)

    path = [start]
    visited = {start}
    state = start
    reached = False
    burned = False
    total = 0.0
    for t in range(max_steps):
        # With time in the state, stepping back onto a node is a legal move,
        # and an under-trained table will oscillate. The fire only grows, so a
        # revisit can never improve an escape; skipping visited nodes keeps the
        # route simple without excluding any route that works.
        k = greedy(state, t, avoid=visited)
        if k is None:
            break
        nxt = target[k]
        path.append(nxt)
        visited.add(nxt)
        if t >= ignite.get(nxt, never):
            burned = True
            total += BURN_REWARD
            break
        if nxt == exit_node:
            reached = True
            total += EXIT_REWARD
            break
        total -= cost[k]
        state = nxt

    points = graph.points[path].astype(np.float64)
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(path) > 1 else 0.0

    escape = earliest_escape(graph, start, exit_node, ignite_time, max_steps, blocked)
    note = None
    if escape is None and not timed:
        note = f"The exit is more than {max_steps} moves away; raise the step limit."
    elif escape is None:
        note = (
            "No route escapes this fire within the step limit: it reaches or cuts off "
            "the exit before any path can."
        )
    elif not reached:
        note = f"An escape exists ({escape} moves) but the policy did not find it; train for more episodes."

    tail = successes[-max(1, len(successes) // 5):]
    return RLResult(
        path=path,
        reached_exit=reached,
        episodes=episodes,
        reward=round(total, 4),
        burned=burned,
        length=length,
        optimal_length=float(distance[start]),
        success_rate=float(np.mean(tail)) if tail else 0.0,
        curve=_bucket(successes),
        escape_steps=escape,
        note=note,
    )


def _bucket(values: List[int]) -> List[float]:
    if not values:
        return []
    chunks = np.array_split(np.asarray(values, dtype=np.float64), min(CURVE_POINTS, len(values)))
    return [round(float(c.mean()), 3) for c in chunks]
