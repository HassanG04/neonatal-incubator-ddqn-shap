#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay_buffer.py -- Capacity-bounded uniform experience replay.

Each transition stores ``(state, action, reward, next_state, terminated,
truncated, scenario)``.  ``terminated`` and ``truncated`` are kept SEPARATE:
only ``terminated`` (a hard-safety breach = true MDP terminal) removes the
bootstrap term from the TD target; a time-limit ``truncated`` transition still
bootstraps from ``next_state``.

Storage is pre-allocated numpy arrays used as a ring buffer (O(1) insert, no
Python object churn).  States are stored **already normalised** by the
agent's :class:`~q_network.ObservationNormalizer`.

>>> SIMULATION-ONLY ACADEMIC PROJECT -- NOT FOR CLINICAL USE. <<<
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

__all__ = ["ReplayBuffer"]


class ReplayBuffer:
    """Uniform-sampling circular replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions; the oldest are overwritten first.
    state_size : int
        Length of each (normalised) state vector.
    seed : int, optional
        Seed of the buffer's private sampling RNG (reproducibility).
    """

    def __init__(self, capacity: int, state_size: int,
                 seed: Optional[int] = None) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if state_size < 1:
            raise ValueError(f"state_size must be >= 1, got {state_size}")
        self.capacity = int(capacity)
        self.state_size = int(state_size)
        self.states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.bool_)
        self.truncated = np.zeros(self.capacity, dtype=np.bool_)
        self.scenarios = np.zeros(self.capacity, dtype=np.int16)
        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- #
    def __len__(self) -> int:
        return self._size

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, terminated: bool, truncated: bool,
            scenario: int = 0) -> None:
        """Insert one transition (overwrites the oldest when full)."""
        i = self._ptr
        self.states[i] = state
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_states[i] = next_state
        self.terminated[i] = bool(terminated)
        self.truncated[i] = bool(truncated)
        self.scenarios[i] = int(scenario)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return 0 < batch_size <= self._size

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Uniformly sample ``batch_size`` transitions (without replacement).

        Raises
        ------
        ValueError
            If the buffer holds fewer than ``batch_size`` transitions or
            ``batch_size`` < 1 (callers should check :meth:`can_sample`).
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if batch_size > self._size:
            raise ValueError(
                f"cannot sample {batch_size} transitions from a buffer of "
                f"size {self._size}")
        idx = self._rng.choice(self._size, size=batch_size, replace=False)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "terminated": self.terminated[idx],
            "truncated": self.truncated[idx],
            "scenarios": self.scenarios[idx],
        }

    def scenario_counts(self) -> Dict[int, int]:
        """Number of stored transitions per scenario id (balance check)."""
        vals, counts = np.unique(self.scenarios[: self._size], return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}
