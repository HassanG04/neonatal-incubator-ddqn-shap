#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_utils.py -- Shared helpers for DDQN training and controller evaluation.

* Deterministic seeding of Python / NumPy / PyTorch.
* The **seed partition** that keeps training, validation and final test
  episodes disjoint (hyper-parameters / checkpoints are selected on
  VALIDATION seeds only; TEST seeds are touched only by
  ``evaluate_controllers.py``).
* Balanced scenario sampling for multi-scenario training.
* A single episode runner + metric computation shared by validation during
  training and by the final evaluation, so every controller (rule-based
  baseline, DDQN, DQN) is measured with the SAME code.
* Run-manifest generation (library versions, device, file hashes, config).

>>> SIMULATION-ONLY ACADEMIC PROJECT -- NOT FOR CLINICAL USE. <<<
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "set_global_seeds", "train_seed", "validation_seed", "test_seed",
    "ScenarioSampler", "Policy", "BaselinePolicy", "AgentPolicy",
    "run_episode", "episode_metrics", "termination_cause",
    "build_manifest", "write_json", "PROJECT_ROOT", "SRC_DIR",
]

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# --------------------------------------------------------------------------- #
#  Seeding                                                                    #
# --------------------------------------------------------------------------- #
def set_global_seeds(seed: int, deterministic_tf: bool = True) -> None:
    """Seed Python, NumPy and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass
        tf.random.set_seed(int(seed))
    except ImportError:  # pragma: no cover
        pass


# Seed partition -- the three ranges never overlap, and none overlaps the
# legacy baseline seeds (42 + scenario*1000 + episode  ->  1042..5046).
#   train      : 1_000_000 * (run_seed + 1) + episode_index
#   validation :   500_000 + scenario * 1000 + i
#   test       :   900_000 + scenario * 1000 + i
def train_seed(run_seed: int, episode_index: int) -> int:
    return 1_000_000 * (int(run_seed) + 1) + int(episode_index)


def validation_seed(scenario: int, i: int) -> int:
    return 500_000 + int(scenario) * 1000 + int(i)


def test_seed(scenario: int, i: int) -> int:
    return 900_000 + int(scenario) * 1000 + int(i)


class ScenarioSampler:
    """Balanced scenario sampling: every block of ``len(scenarios)`` episodes
    contains each scenario exactly once (in a shuffled order).  This keeps
    the replay buffer from being dominated by one scenario even though
    episode lengths differ (early terminations are short)."""

    def __init__(self, scenarios: Sequence[int], seed: Optional[int] = None,
                 mode: str = "balanced") -> None:
        if not scenarios:
            raise ValueError("scenarios must be non-empty")
        if mode not in ("balanced", "uniform"):
            raise ValueError(f"unknown scenario sampling mode {mode!r}")
        self.scenarios = [int(s) for s in scenarios]
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self._queue: List[int] = []

    def next(self) -> int:
        if self.mode == "uniform":
            return int(self.rng.choice(self.scenarios))
        if not self._queue:
            self._queue = list(self.rng.permutation(self.scenarios))
        return int(self._queue.pop())


# --------------------------------------------------------------------------- #
#  Policies (common interface for evaluation)                                 #
# --------------------------------------------------------------------------- #
class Policy:
    name: str = "policy"

    def reset(self, env) -> None:  # called once per episode, after env.reset
        pass

    def act(self, obs: np.ndarray) -> int:  # pragma: no cover - interface
        raise NotImplementedError


class BaselinePolicy(Policy):
    """Rule-based PID/bang-bang baseline, grid matched to each env."""

    def __init__(self, name: str = "baseline") -> None:
        self.name = name
        self._ctrl = None

    def reset(self, env) -> None:
        from baseline import RuleBasedBaselineController
        self._ctrl = RuleBasedBaselineController.from_env(env)
        self._ctrl.reset()

    def act(self, obs: np.ndarray) -> int:
        return int(self._ctrl.select_action(obs))


class AgentPolicy(Policy):
    """Greedy (epsilon = 0) policy of a trained DDQN/DQN agent."""

    def __init__(self, agent, name: str = "ddqn") -> None:
        self.agent = agent
        self.name = name

    def reset(self, env) -> None:
        if env.action_space.n != self.agent.action_size:
            raise ValueError("agent/env action-space mismatch")
        if env.observation_space.shape[0] != self.agent.state_size:
            raise ValueError("agent/env observation-space mismatch")

    def act(self, obs: np.ndarray) -> int:
        return self.agent.greedy_action(self.agent.preprocess(obs))


# --------------------------------------------------------------------------- #
#  Episode runner + metrics                                                   #
# --------------------------------------------------------------------------- #
def termination_cause(info: Dict[str, Any], terminated: bool,
                      truncated: bool) -> str:
    if terminated:
        for key in ("hypothermia", "hyperthermia", "hypoxia", "hyperoxia"):
            if info.get(key):
                return key
        return "safety_breach"
    if truncated:
        return "max_steps"
    return "in_progress"


def run_episode(env, policy: Policy, seed: int,
                record_trajectory: bool = True) -> Dict[str, Any]:
    """Run one full episode; return metrics (+ optional per-step trajectory).

    The episode stops on ``terminated or truncated`` (Gymnasium contract).
    No auto-reset wrapper is used, so the final observation is genuine.
    """
    obs, info = env.reset(seed=int(seed))
    policy.reset(env)
    rows: List[Dict[str, Any]] = []
    rows.append(_traj_row(0, -1, obs, info, 0.0, False, False))
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action = int(policy.act(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        rows.append(_traj_row(step, action, obs, info, float(reward),
                              terminated, truncated))
    metrics = episode_metrics(rows, env)
    metrics["termination_cause"] = termination_cause(info, terminated, truncated)
    metrics["seed"] = int(seed)
    metrics["scenario"] = int(env.scenario)
    out: Dict[str, Any] = {"metrics": metrics}
    if record_trajectory:
        out["trajectory"] = rows
    return out


def _traj_row(step, action, obs, info, reward, terminated, truncated):
    return {
        "step": step, "action": action,
        "heater_idx": info.get("heater_idx", -1) if action >= 0 else -1,
        "o2_idx": info.get("o2_idx", -1) if action >= 0 else -1,
        "u_heater": info.get("u_heater", np.nan) if action >= 0 else np.nan,
        "fio2_cmd": info.get("fio2_cmd", np.nan) if action >= 0 else np.nan,
        "Q_heater_W": info.get("Q_heater_W", np.nan) if action >= 0 else np.nan,
        "Tc": float(info["Tc"]), "Ta": float(info["Ta"]),
        "C_O2": float(info["C_O2"]), "SpO2": float(info["SpO2"]),
        "Tamb": float(info["Tamb"]), "disturbance": float(info["disturbance"]),
        "reward": reward,
        "r_thermal": info.get("r_thermal", 0.0), "r_oxygen": info.get("r_oxygen", 0.0),
        "r_energy": info.get("r_energy", 0.0), "r_safety": info.get("r_safety", 0.0),
        "terminated": bool(terminated), "truncated": bool(truncated),
    }


def _recovery_time(values: np.ndarray, in_band: np.ndarray, start: int,
                   hold: int = 5) -> float:
    """Steps from ``start`` until the signal is in band for ``hold`` steps.

    Returns NaN if it never recovers inside the episode."""
    n = len(values)
    for t in range(max(0, start), n - hold + 1):
        if in_band[t:t + hold].all():
            return float(t - start)
    return float("nan")


def episode_metrics(rows: List[Dict[str, Any]], env) -> Dict[str, float]:
    """Compute the evaluation metrics for one episode trajectory.

    Row 0 is the post-reset state (no action); rows 1..N follow actions.
    State metrics use rows 1..N (states the controller produced).
    """
    steps = rows[1:]
    n = len(steps)
    tc = np.array([r["Tc"] for r in steps], dtype=float)
    spo2 = np.array([r["SpO2"] for r in steps], dtype=float)
    u_h = np.array([r["u_heater"] for r in steps], dtype=float)
    fio2 = np.array([r["fio2_cmd"] for r in steps], dtype=float) / 100.0
    actions = np.array([r["action"] for r in steps], dtype=int)
    rewards = np.array([r["reward"] for r in steps], dtype=float)
    tc_err = np.abs(tc - env.tc_setpoint)
    lo, hi = env.spo2_target_low, env.spo2_target_high
    spo2_band_err = np.where(spo2 < lo, lo - spo2, np.where(spo2 > hi, spo2 - hi, 0.0))
    tc_in = (tc >= 36.5) & (tc <= 37.5)
    spo2_in = (spo2 >= lo) & (spo2 <= hi)
    u_fio2 = (fio2 - env.fio2_min) / (env.fio2_max - env.fio2_min)
    terminated = bool(steps[-1]["terminated"]) if n else False

    m: Dict[str, float] = {
        "total_reward": float(rewards.sum()),
        "episode_length": int(n),
        "survived": float(not terminated),
        "safety_violation": float(terminated),
        "tc_mean_abs_err": float(tc_err.mean()) if n else np.nan,
        "tc_max_abs_err": float(tc_err.max()) if n else np.nan,
        "tc_time_in_band": float(tc_in.mean()) if n else np.nan,
        "spo2_mean_band_err": float(spo2_band_err.mean()) if n else np.nan,
        "spo2_max_band_err": float(spo2_band_err.max()) if n else np.nan,
        "spo2_time_in_band": float(spo2_in.mean()) if n else np.nan,
        "mean_heater_frac": float(u_h.mean()) if n else np.nan,
        "mean_fio2_norm": float(u_fio2.mean()) if n else np.nan,
        "control_effort": float((u_h + u_fio2).mean()) if n else np.nan,
        "heater_energy_kJ": float((u_h * env.q_heater_max * env.dt).sum() / 1000.0),
        "action_switches": int((np.diff(actions) != 0).sum()) if n > 1 else 0,
    }

    # Disturbance recovery / overshoot (scenario-specific, NaN where N/A).
    scn = int(env.scenario)
    m["recovery_steps"] = np.nan
    m["overshoot"] = np.nan
    if n:
        if scn == 3:   # thermal disturbance: porthole closes at porthole_end
            start = int(env.porthole_end)
            m["recovery_steps"] = _recovery_time(tc, tc_in, start)
            post = tc[start:] if start < n else np.array([])
            m["overshoot"] = float(max(0.0, (post - 37.5).max())) if post.size else np.nan
        elif scn == 4:  # initial hypothermia: recovery from t=0
            m["recovery_steps"] = _recovery_time(tc, tc_in, 0)
            m["overshoot"] = float(max(0.0, (tc - 37.5).max()))
        elif scn == 5:  # desaturation: from event onset
            start = int(env.o2_disturb_start)
            m["recovery_steps"] = _recovery_time(spo2, spo2_in, start)
            post = spo2[start:] if start < n else np.array([])
            m["overshoot"] = float(max(0.0, (post - hi).max())) if post.size else np.nan
        elif scn == 2:  # slow ambient drift: worst-case Tc error in last half
            half = tc_err[n // 2:]
            m["overshoot"] = float(half.max()) if half.size else np.nan
    return m


# --------------------------------------------------------------------------- #
#  Manifest                                                                   #
# --------------------------------------------------------------------------- #
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_revision() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # pragma: no cover
        pass
    return None


def build_manifest(config: Dict[str, Any], seeds: Dict[str, Any],
                   device: str, results: Optional[Dict[str, Any]] = None,
                   config_path: Optional[str] = None) -> Dict[str, Any]:
    """Machine-readable run manifest (versions, hashes, device, seeds...)."""
    versions = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "tensorflow", "gymnasium", "yaml", "matplotlib"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    gpu = None
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            details = tf.config.experimental.get_device_details(gpus[0])
            gpu = details.get("device_name", gpus[0].name)
    except Exception:  # pragma: no cover
        pass
    src_hashes = {f: _sha256(os.path.join(SRC_DIR, f))
                  for f in sorted(os.listdir(SRC_DIR)) if f.endswith(".py")}
    cfg_file = config_path or os.path.join(PROJECT_ROOT, "config.yaml")
    return {
        "created_unix": time.time(),
        "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_revision": _git_revision(),
        "source_sha256": src_hashes,
        "config_file": os.path.relpath(cfg_file, PROJECT_ROOT) if os.path.isfile(cfg_file) else None,
        "config_sha256": _sha256(cfg_file) if os.path.isfile(cfg_file) else None,
        "config": config,
        "seeds": seeds,
        "device": device,
        "gpu": gpu,
        "platform": platform.platform(),
        "versions": versions,
        "results": results or {},
        "disclaimer": "Simulation-only academic project; not clinically validated.",
    }


def write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)
