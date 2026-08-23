"""Tabular Q-learning over the navigation graph.

Same algorithm as Classic, with two corrections that changed the output:

* the greedy rollout used to stop the moment it revisited a node but still
  appended it, so a trapped policy returned a path with a dangling repeat.
  It now stops cleanly and reports whether the exit was actually reached.
* ties in ``max(neighbours, key=q)`` were resolved by list order, which made
  the result depend on adjacency construction order. Ties now break randomly
  under the caller's seed, so a run is reproducible but not biased.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..store import NavGraph


@dataclass
class RLResult:
    path: List[int]
    reached_exit: bool
    episodes: int
    reward: float


def q_learning_path(
    graph: NavGraph,
    start: int,
    exit_node: int,
    ignite_time: Optional[Dict[int, int]] = None,
    episodes: int = 200,
    max_steps: int = 200,
    epsilon: float = 0.2,
    alpha: float = 0.5,
    gamma: float = 0.9,
    seed: Optional[int] = None,
) -> RLResult:
    n = graph.node_count
    if n == 0 or not (0 <= start < n) or not (0 <= exit_node < n):
        return RLResult([], False, 0, 0.0)
    if start == exit_node:
        return RLResult([start], True, 0, 0.0)

    rng = random.Random(seed)
    indptr, neighbours, _ = graph.csr()

    def neighbours_of(node: int) -> np.ndarray:
        return neighbours[indptr[node] : indptr[node + 1]]

    q: Dict[int, Dict[int, float]] = {}

    def qrow(node: int) -> Dict[int, float]:
        row = q.get(node)
        if row is None:
            row = {}
            q[node] = row
        return row

    def best(node: int) -> Optional[int]:
        options = neighbours_of(node)
        if len(options) == 0:
            return None
        row = qrow(node)
        top = -float("inf")
        winners: List[int] = []
        for nbr in options:
            nbr = int(nbr)
            value = row.get(nbr, 0.0)
            if value > top:
                top = value
                winners = [nbr]
            elif value == top:
                winners.append(nbr)
        return rng.choice(winners)

    for _ in range(max(1, episodes)):
        state = start
        for t in range(max_steps):
            options = neighbours_of(state)
            if len(options) == 0:
                break
            if rng.random() < epsilon:
                nxt = int(options[rng.randrange(len(options))])
            else:
                chosen = best(state)
                if chosen is None:
                    break
                nxt = chosen

            reward = -0.1
            done = False
            if ignite_time is not None and t >= ignite_time.get(nxt, 1 << 30):
                reward = -10.0
                done = True
            if nxt == exit_node:
                reward = 10.0
                done = True

            next_row = qrow(nxt)
            max_next = max(next_row.values()) if next_row else 0.0
            row = qrow(state)
            row[nxt] = (1 - alpha) * row.get(nxt, 0.0) + alpha * (reward + gamma * max_next)

            state = nxt
            if done:
                break

    path = [start]
    visited = {start}
    state = start
    reached = False
    total_reward = 0.0
    for _ in range(max_steps):
        nxt = best(state)
        if nxt is None or nxt in visited:
            break
        path.append(nxt)
        visited.add(nxt)
        total_reward += qrow(state).get(nxt, 0.0)
        state = nxt
        if nxt == exit_node:
            reached = True
            break

    return RLResult(path=path, reached_exit=reached, episodes=episodes, reward=total_reward)
