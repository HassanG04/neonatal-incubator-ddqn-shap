#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ddqn_agent.py -- Double Deep Q-Network agent (with a standard-DQN switch)
implemented with TensorFlow / Keras (tf.keras).

Core update (van Hasselt, Guez & Silver 2016, arXiv:1509.06461)::

    a*  = argmax_a Q_online(s', a)                      # online net SELECTS
    y   = r + gamma * (1 - terminated) * Q_target(s', a*)   # target net EVALUATES
    L   = Huber( Q_online(s, a) , y )

* ``terminated`` (hard-safety breach) is the ONLY flag that removes the
  bootstrap.  A time-limit ``truncated`` transition still bootstraps, because
  the episode was cut by the step budget, not by an MDP terminal state.
* Target values are computed with ``tf.stop_gradient``; the target network
  is never passed to the optimiser.
* ``double=False`` turns the agent into standard DQN
  (``y = r + gamma*(1-terminated)*max_a Q_target(s', a)``) -- used only as an
  algorithmic comparison.

>>> SIMULATION-ONLY ACADEMIC PROJECT -- NOT FOR CLINICAL USE. <<<
"""
from __future__ import annotations

import os
import pickle
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from q_network import ObservationNormalizer, QNetwork

__all__ = [
    "AgentConfig", "DDQNAgent", "LinearEpsilonSchedule",
    "double_dqn_targets", "dqn_targets", "select_device",
]


# --------------------------------------------------------------------------- #
#  Pure target functions (unit-tested independently of the agent)             #
# --------------------------------------------------------------------------- #
def double_dqn_targets(rewards: tf.Tensor, next_q_online: tf.Tensor,
                       next_q_target: tf.Tensor, terminated: tf.Tensor,
                       gamma: float) -> tf.Tensor:
    """Double-DQN TD target.

    Parameters
    ----------
    rewards : (B,) float
    next_q_online : (B, A) -- Q_online(s', .)  (used only for argmax)
    next_q_target : (B, A) -- Q_target(s', .)  (used to evaluate the argmax)
    terminated : (B,) bool/float -- 1 => no bootstrap
    gamma : float
    """
    rewards_t = tf.cast(rewards, tf.float32)
    next_q_online_t = tf.cast(next_q_online, tf.float32)
    next_q_target_t = tf.cast(next_q_target, tf.float32)
    term_t = tf.cast(terminated, tf.float32)

    best_next = tf.argmax(next_q_online_t, axis=1, output_type=tf.int32)  # select
    batch_idx = tf.range(tf.shape(next_q_target_t)[0], dtype=tf.int32)
    gather_idx = tf.stack([batch_idx, best_next], axis=1)
    next_val = tf.gather_nd(next_q_target_t, gather_idx)                  # evaluate
    not_done = 1.0 - term_t
    return tf.stop_gradient(rewards_t + tf.cast(gamma, tf.float32) * not_done * next_val)


def dqn_targets(rewards: tf.Tensor, next_q_target: tf.Tensor,
                terminated: tf.Tensor, gamma: float) -> tf.Tensor:
    """Standard (Nature) DQN TD target: max over the target network."""
    rewards_t = tf.cast(rewards, tf.float32)
    next_q_target_t = tf.cast(next_q_target, tf.float32)
    term_t = tf.cast(terminated, tf.float32)

    next_val = tf.reduce_max(next_q_target_t, axis=1)
    not_done = 1.0 - term_t
    return tf.stop_gradient(rewards_t + tf.cast(gamma, tf.float32) * not_done * next_val)


def select_device(preference: str = "auto") -> str:
    """``auto`` -> CPU for tiny MLP (avoids host-device transfer overhead per step)
    or honours explicit ``cpu`` / ``gpu`` / ``cuda`` requests."""
    pref = str(preference).lower()
    if pref in ("cpu", "/cpu:0", "auto"):
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        return "/CPU:0"
    gpus = tf.config.list_physical_devices("GPU")
    if pref in ("cuda", "gpu", "/gpu:0", "cuda:0", "gpu:0"):
        if not gpus:
            raise RuntimeError("GPU/CUDA requested but no TensorFlow GPU is available")
        return "/GPU:0"
    return str(preference)


# --------------------------------------------------------------------------- #
#  Exploration schedule                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class LinearEpsilonSchedule:
    """Linear decay from ``start`` to ``end`` over ``decay_steps`` env steps,
    held at ``start`` for the first ``warmup_steps`` steps."""
    start: float = 1.0
    end: float = 0.05
    decay_steps: int = 100_000
    warmup_steps: int = 0

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(self.start)
        frac = min(1.0, (step - self.warmup_steps) / max(1, self.decay_steps))
        return float(self.start + frac * (self.end - self.start))


# --------------------------------------------------------------------------- #
#  Agent                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class AgentConfig:
    gamma: float = 0.99
    lr: float = 3.0e-4
    batch_size: int = 128
    hidden_sizes: Tuple[int, ...] = (128, 128)
    grad_clip_norm: float = 10.0
    target_update: str = "hard"          # "hard" | "soft"
    target_update_interval: int = 500    # gradient steps between hard copies
    tau: float = 0.005                   # soft-update coefficient
    huber_delta: float = 1.0
    double: bool = True                  # False => standard DQN target

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "hidden_sizes" in known:
            known["hidden_sizes"] = tuple(known["hidden_sizes"])
        return cls(**known)


def _smooth_l1_loss(q_taken: tf.Tensor, targets: tf.Tensor, delta: Any) -> tf.Tensor:
    """Elementwise Huber / smooth-L1 loss matching PyTorch F.smooth_l1_loss(beta=delta)."""
    diff = tf.abs(q_taken - targets)
    beta = tf.cast(delta, tf.float32)
    quadratic = tf.minimum(diff, beta)
    linear = diff - quadratic
    loss_per_sample = 0.5 * tf.square(quadratic) / beta + linear
    return tf.reduce_mean(loss_per_sample)


class DDQNAgent:
    """Double DQN agent with online/target Q-networks in TensorFlow / Keras.

    Parameters
    ----------
    state_size, action_size : int
        Inferred at runtime from the environment.
    normalizer : ObservationNormalizer
        Applied to raw observations in :meth:`preprocess`; saved in every
        checkpoint so evaluation uses the identical transform.
    config : AgentConfig
    device : str
        TensorFlow device specification (``"auto"``, ``"cpu"``, ``"cuda"`` / ``"gpu"``).
    seed : int, optional
        Seeds the agent's private exploration RNG and weight initialization.
    """

    def __init__(self, state_size: int, action_size: int,
                 normalizer: ObservationNormalizer,
                 config: Optional[AgentConfig] = None,
                 device: Any = "cpu", seed: Optional[int] = None) -> None:
        self.config = config or AgentConfig()
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        if normalizer.state_size != self.state_size:
            raise ValueError(
                f"normalizer has {normalizer.state_size} features, "
                f"state_size={self.state_size}")
        self.normalizer = normalizer
        self.device = select_device(str(device))
        with tf.device(self.device):
            self.online = QNetwork(
                state_size, action_size, self.config.hidden_sizes, seed=seed
            )
            self.target = QNetwork(
                state_size, action_size, self.config.hidden_sizes, seed=seed
            )
            self.target.set_weights(self.online.get_weights())
            self.target.trainable = False
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.lr)

        self.rng = np.random.default_rng(seed)
        self.grad_steps = 0
        self.target_syncs = 0
        self.loaded_extra: Dict[str, Any] = {}
        self._compiled_train_step = tf.function(
            self._train_step_impl, reduce_retracing=True
        )

    # ---------------------------------------------------------------- #
    #  Acting                                                           #
    # ---------------------------------------------------------------- #
    def preprocess(self, obs: np.ndarray) -> np.ndarray:
        return self.normalizer(obs)

    def q_values(self, norm_state: np.ndarray) -> np.ndarray:
        """Q-values for one (already normalised) state or a batch."""
        x = np.atleast_2d(np.asarray(norm_state, dtype=np.float32))
        if x.shape[0] <= 16:
            return self.online.forward_np(x)
        with tf.device(self.device):
            out = self.online(tf.constant(x, dtype=tf.float32), training=False)
        return out.numpy()

    def greedy_action(self, norm_state: np.ndarray) -> int:
        """Deterministic argmax action (evaluation mode, epsilon = 0)."""
        return int(np.argmax(self.q_values(norm_state)[0]))

    def act(self, norm_state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action restricted to ``0..action_size-1``."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_size))
        return self.greedy_action(norm_state)

    # ---------------------------------------------------------------- #
    #  Learning                                                         #
    # ---------------------------------------------------------------- #
    def _to_tensors(self, batch: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:
        with tf.device(self.device):
            return {
                "states": tf.convert_to_tensor(batch["states"], dtype=tf.float32),
                "actions": tf.convert_to_tensor(batch["actions"], dtype=tf.int32),
                "rewards": tf.convert_to_tensor(batch["rewards"], dtype=tf.float32),
                "next_states": tf.convert_to_tensor(batch["next_states"], dtype=tf.float32),
                "terminated": tf.convert_to_tensor(batch["terminated"], dtype=tf.float32),
            }

    def compute_targets(self, rewards: tf.Tensor, next_states: tf.Tensor,
                        terminated: tf.Tensor) -> tf.Tensor:
        """TD targets (Double DQN, or standard DQN if ``config.double=False``)."""
        with tf.device(self.device):
            next_q_target = self.target(next_states, training=False)
            if self.config.double:
                next_q_online = self.online(next_states, training=False)
                return double_dqn_targets(rewards, next_q_online, next_q_target,
                                          terminated, self.config.gamma)
            return dqn_targets(rewards, next_q_target, terminated, self.config.gamma)

    def _train_step_impl(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        terminated: tf.Tensor,
        gamma: tf.Tensor,
        huber_delta: tf.Tensor,
        grad_clip_norm: tf.Tensor,
        is_double: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        next_q_target = self.target(next_states, training=False)
        if is_double:
            next_q_online = self.online(next_states, training=False)
            targets = double_dqn_targets(rewards, next_q_online, next_q_target,
                                         terminated, gamma)
        else:
            targets = dqn_targets(rewards, next_q_target, terminated, gamma)

        batch_idx = tf.range(tf.shape(states)[0], dtype=tf.int32)
        gather_idx = tf.stack([batch_idx, actions], axis=1)

        with tf.GradientTape() as tape:
            q_all = self.online(states, training=True)
            q_taken = tf.gather_nd(q_all, gather_idx)
            loss = _smooth_l1_loss(q_taken, targets, huber_delta)

        grads = tape.gradient(loss, self.online.trainable_variables)
        clipped_grads, grad_norm = tf.clip_by_global_norm(grads, grad_clip_norm)
        # Apply gradients only when loss is finite so a non-finite batch raises
        # FloatingPointError without corrupting weights.
        if tf.math.is_finite(loss):
            self.optimizer.apply_gradients(
                zip(clipped_grads, self.online.trainable_variables)
            )

        td_abs = tf.abs(targets - q_taken)
        return (
            loss,
            tf.reduce_mean(q_taken),
            tf.reduce_max(q_all),
            tf.reduce_mean(targets),
            tf.reduce_mean(td_abs),
            tf.reduce_max(td_abs),
            grad_norm,
        )

    def learn(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """One gradient step on a replay minibatch; returns diagnostics.

        Raises ``FloatingPointError`` if the loss is not finite (fail fast
        rather than silently corrupting the network).
        """
        t = self._to_tensors(batch)
        with tf.device(self.device):
            (
                loss,
                q_mean,
                q_max,
                target_mean,
                td_abs_mean,
                td_abs_max,
                grad_norm,
            ) = self._compiled_train_step(
                t["states"],
                t["actions"],
                t["rewards"],
                t["next_states"],
                t["terminated"],
                tf.constant(float(self.config.gamma), dtype=tf.float32),
                tf.constant(float(self.config.huber_delta), dtype=tf.float32),
                tf.constant(float(self.config.grad_clip_norm), dtype=tf.float32),
                bool(self.config.double),
            )

        loss_val = float(loss.numpy())
        if not np.isfinite(loss_val):
            raise FloatingPointError(f"non-finite loss {loss_val}")

        self.online.invalidate_np_cache()
        self.grad_steps += 1
        self._maybe_update_target()

        return {
            "loss": loss_val,
            "q_mean": float(q_mean.numpy()),
            "q_max": float(q_max.numpy()),
            "target_mean": float(target_mean.numpy()),
            "td_abs_mean": float(td_abs_mean.numpy()),
            "td_abs_max": float(td_abs_max.numpy()),
            "grad_norm": float(grad_norm.numpy()),
        }

    def _maybe_update_target(self) -> None:
        cfg = self.config
        if cfg.target_update == "soft":
            self.soft_update(cfg.tau)
        elif cfg.target_update == "hard":
            if self.grad_steps % max(1, cfg.target_update_interval) == 0:
                self.hard_update()
        else:
            raise ValueError(f"unknown target_update {cfg.target_update!r}")

    def hard_update(self) -> None:
        for tp, op in zip(self.target.variables, self.online.variables):
            tp.assign(op)
        self.target_syncs += 1

    def soft_update(self, tau: float) -> None:
        tau_f = float(tau)
        for tp, op in zip(self.target.variables, self.online.variables):
            tp.assign((1.0 - tau_f) * tp + tau_f * op)
        self.target_syncs += 1

    # ---------------------------------------------------------------- #
    #  Checkpointing                                                    #
    # ---------------------------------------------------------------- #
    def checkpoint(self, extra: Optional[Dict[str, Any]] = None,
                   include_optimizer: bool = True) -> Dict[str, Any]:
        ckpt: Dict[str, Any] = {
            "format": "neonatal-ddqn-tf-v1",
            "backend": "tensorflow",
            "tf_version": tf.__version__,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "agent_config": self.config.to_dict(),
            "normalizer": self.normalizer.state_dict(),
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "grad_steps": self.grad_steps,
            "target_syncs": self.target_syncs,
            "extra": extra or {},
        }
        if include_optimizer:
            try:
                ckpt["optimizer"] = [
                    np.array(v.numpy(), copy=True) for v in self.optimizer.variables
                ]
            except Exception:
                ckpt["optimizer"] = []
        return ckpt

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None,
             include_optimizer: bool = True) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                self.checkpoint(extra, include_optimizer),
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str, device: Any = "cpu",
             seed: Optional[int] = None) -> "DDQNAgent":
        """Rebuild an agent (networks, normaliser, optimiser) from ``path``."""
        with open(path, "rb") as fh:
            ckpt = pickle.load(fh)
        agent = cls(ckpt["state_size"], ckpt["action_size"],
                    ObservationNormalizer.from_state_dict(ckpt["normalizer"]),
                    AgentConfig.from_dict(ckpt["agent_config"]),
                    device=device, seed=seed)
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["target"])
        opt_vars = ckpt.get("optimizer")
        if opt_vars:
            try:
                if hasattr(agent.optimizer, "build"):
                    agent.optimizer.build(agent.online.trainable_variables)
                if len(agent.optimizer.variables) == len(opt_vars):
                    for var, val in zip(agent.optimizer.variables, opt_vars):
                        var.assign(val)
            except Exception:
                pass
        agent.grad_steps = int(ckpt.get("grad_steps", 0))
        agent.target_syncs = int(ckpt.get("target_syncs", 0))
        agent.loaded_extra = ckpt.get("extra", {})
        return agent
