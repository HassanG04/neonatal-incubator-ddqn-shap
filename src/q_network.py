#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q_network.py -- Q-network and observation pre-processing for DDQN / DQN
implemented with TensorFlow / Keras (tf.keras).

Contents
--------
* :class:`QNetwork` -- feed-forward MLP ``state_size -> 128 -> 128 -> action_size``
  with ReLU hidden activations and a **linear** Q-value output (no softmax /
  sigmoid: Q-values are unbounded real-valued returns).
* :class:`ObservationNormalizer` -- fixed, *predetermined* affine scaling of
  the mixed-unit observation (deg C, %, deg C/s, %/s, phase, flags) to a
  roughly unit range.  The transform is part of every checkpoint so the
  **same** scaling is applied during training, evaluation and inference.

Why predetermined bounds instead of running statistics?
    * They are known before any data is collected (no train/test leakage,
      no dependency on the exploration policy that generated the data).
    * They are deterministic and identical across seeds.
    * The raw ``observation_space`` bounds are far wider than the operating
      region (e.g. Tc in [30, 42] while the safe band is [35.5, 38]); a
      min-max on those bounds would squeeze the whole controllable region of
      Tc into ~20% of the input range.  We therefore centre each feature on
      its nominal operating point and scale by a physically motivated
      half-range (see ``DEFAULT_SCALING``).  Features not in the table fall
      back to the observation-space bounds.

>>> SIMULATION-ONLY ACADEMIC PROJECT -- NOT FOR CLINICAL USE. <<<
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

__all__ = ["QNetwork", "ObservationNormalizer", "DEFAULT_SCALING"]


# (centre, half-range) per observation feature, in raw units.
# normalised = clip((x - centre) / half_range, -clip, +clip)
DEFAULT_SCALING: Dict[str, Tuple[float, float]] = {
    "Tc":          (37.0, 1.5),    # deg C; safe band 35.5..38.0 -> [-1, +0.67]
    "Tsk":         (36.5, 1.5),    # deg C; Tsk = Tc - 0.5
    "Ta":          (34.0, 6.0),    # deg C; air node clipped to [20, 40]
    "C_O2":        (40.5, 19.5),   # %; inspired O2 in [21, 60] -> [-1, +1]
    "SpO2":        (92.0, 6.0),    # %; hard band 85..96 -> [-1.17, +0.67]
    "Tamb":        (22.0, 5.0),    # deg C; 25 nominal, drops to ~18 in scn 2
    "dTa_dt":      (0.0, 0.05),    # deg C/s; ~ +/-2.25 deg C per 45 s step
    "dC_O2_dt":    (0.0, 0.5),     # %/s; ~ +/-22 % per 45 s step
    "phase":       (0.5, 0.5),     # [0, 1] -> [-1, +1]
    "disturbance": (1.0, 1.0),     # {0, 1, 2} -> {-1, 0, +1}
    "scenario_id": (0.5, 0.5),     # [0, 1] -> [-1, +1]
}


class ObservationNormalizer:
    """Deterministic affine observation scaling, serialisable to a dict.

    Parameters
    ----------
    center, scale : array-like, shape (state_size,)
        ``normalised = clip((obs - center) / scale, -clip_value, clip_value)``.
    clip_value : float
        Symmetric clip applied after scaling (guards against out-of-range
        inputs producing extreme activations).
    names : sequence of str, optional
        Feature names, stored for traceability only.
    """

    def __init__(self, center: Sequence[float], scale: Sequence[float],
                 clip_value: float = 5.0,
                 names: Optional[Sequence[str]] = None) -> None:
        self.center = np.asarray(center, dtype=np.float32).reshape(-1)
        self.scale = np.asarray(scale, dtype=np.float32).reshape(-1)
        if self.center.shape != self.scale.shape:
            raise ValueError("center and scale must have the same shape")
        if np.any(self.scale <= 0) or not np.all(np.isfinite(self.scale)):
            raise ValueError(f"scale must be finite and > 0, got {self.scale}")
        self.clip_value = float(clip_value)
        self.names = list(names) if names is not None else None

    # ---------------------------------------------------------------- #
    @classmethod
    def from_env(cls, env, clip_value: float = 5.0,
                 scaling: Optional[Dict[str, Tuple[float, float]]] = None
                 ) -> "ObservationNormalizer":
        """Build the transform for ``env``'s observation layout.

        Uses ``DEFAULT_SCALING`` for every known feature name and falls back
        to the observation-space mid-point / half-width for unknown ones.
        """
        scaling = dict(DEFAULT_SCALING if scaling is None else scaling)
        space = env.observation_space
        n = int(space.shape[0])
        names = list(getattr(env, "obs_names", [f"f{i}" for i in range(n)]))
        if len(names) != n:
            names = [f"f{i}" for i in range(n)]
        low = np.asarray(space.low, dtype=np.float64)
        high = np.asarray(space.high, dtype=np.float64)
        center = np.empty(n, dtype=np.float32)
        scale = np.empty(n, dtype=np.float32)
        for i, name in enumerate(names):
            if name in scaling:
                center[i], scale[i] = scaling[name]
            else:
                center[i] = 0.5 * (low[i] + high[i])
                scale[i] = max(0.5 * (high[i] - low[i]), 1e-6)
        return cls(center, scale, clip_value=clip_value, names=names)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        out = (obs - self.center) / self.scale
        return np.clip(out, -self.clip_value, self.clip_value).astype(np.float32)

    @property
    def state_size(self) -> int:
        return int(self.center.shape[0])

    def state_dict(self) -> dict:
        return {"center": self.center.tolist(), "scale": self.scale.tolist(),
                "clip_value": self.clip_value, "names": self.names}

    @classmethod
    def from_state_dict(cls, d: dict) -> "ObservationNormalizer":
        return cls(d["center"], d["scale"], clip_value=d.get("clip_value", 5.0),
                   names=d.get("names"))


class QNetwork(tf.keras.Model):
    """Feed-forward action-value network ``Q(s, .)`` in TensorFlow / Keras.

    ``state_size -> hidden[0] -> ... -> hidden[-1] -> action_size`` with ReLU
    between hidden layers and a LINEAR output layer (``activation=None``).

    Parameters
    ----------
    state_size : int
        Observation dimension (default env: 11). Inferred at runtime from
        ``env.observation_space.shape[0]``, never hard-coded.
    action_size : int
        Number of discrete actions (default env: 25), from
        ``env.action_space.n``.
    hidden_sizes : iterable of int, default (128, 128)
    seed : int, optional
        Seed for kernel initializers.
    """

    def __init__(self, state_size: int, action_size: int,
                 hidden_sizes: Iterable[int] = (128, 128),
                 seed: Optional[int] = None) -> None:
        super().__init__()
        if state_size < 1 or action_size < 1:
            raise ValueError("state_size and action_size must be >= 1")
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self._np_cache: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

        layers_list: List[tf.keras.layers.Layer] = []
        for idx, h in enumerate(self.hidden_sizes):
            init = tf.keras.initializers.GlorotUniform(
                seed=None if seed is None else int(seed) + idx * 101
            )
            layers_list.append(
                tf.keras.layers.Dense(
                    h,
                    activation="relu",
                    kernel_initializer=init,
                    bias_initializer="zeros",
                    dtype="float32",
                    name=f"hidden_{idx}",
                )
            )
        out_init = tf.keras.initializers.GlorotUniform(
            seed=None if seed is None else int(seed) + 999
        )
        self.output_layer = tf.keras.layers.Dense(
            self.action_size,
            activation=None,  # linear Q-value output (unbounded real values)
            kernel_initializer=out_init,
            bias_initializer="zeros",
            dtype="float32",
            name="q_values",
        )
        layers_list.append(self.output_layer)
        self.net = tf.keras.Sequential(layers_list, name="q_mlp")
        # Build weights immediately with a dummy pass so get_weights() /
        # trainable_variables are always populated before the first call.
        dummy = tf.zeros((1, self.state_size), dtype=tf.float32)
        _ = self.net(dummy)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x_t = tf.cast(x, tf.float32)
        return self.net(x_t, training=training)

    def invalidate_np_cache(self) -> None:
        self._np_cache = None

    def _get_np_cache(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self._np_cache is None:
            self._np_cache = [
                (w.numpy(), b.numpy())
                for w, b in (
                    (layer.kernel, layer.bias) for layer in self.net.layers
                )
            ]
        return self._np_cache

    def forward_np(self, x: np.ndarray) -> np.ndarray:
        """Fast NumPy forward pass for single-state inference (avoids eager overhead)."""
        h = np.asarray(x, dtype=np.float32)
        pairs = self._get_np_cache()
        for w, b in pairs[:-1]:
            h = np.maximum(0.0, h @ w + b)
        w_out, b_out = pairs[-1]
        return (h @ w_out + b_out).astype(np.float32)

    def state_dict(self) -> List[np.ndarray]:
        return [np.array(w, dtype=np.float32, copy=True) for w in self.get_weights()]

    def load_state_dict(self, weights: Sequence[np.ndarray]) -> None:
        self.set_weights([np.asarray(w, dtype=np.float32) for w in weights])
        self._np_cache = None
