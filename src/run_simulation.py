#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 run_simulation.py -- Member 1 baseline reference generator
------------------------------------------------------------------------------
Project : Project 04 -- Neonatal Incubator Regulation (RL + XAI)
          Member 1 deliverable: Simulation Environment, Dataset & MDP.
Module  : run_simulation.py
Purpose : Execute the rule-based (PID + rule-based FiO2) baseline controller
          across every experimental scenario defined in
          :class:`environment.NeonatalIncubatorEnv`, collect per-step
          trajectories and per-episode summary statistics, and persist them
          as CSV files under ``--output-dir`` (default ``results``).

ROLE IN MEMBER 1'S DELIVERABLES
------------------------------
This script is the **baseline reference generator** for Member 1. It
produces the reference numbers against which the DDQN agent (developed
by Member 2) will be compared, and against which SHAP attributions
(Member 3) will be interpreted. **Member 2's DDQN must beat these
numbers** on at least four of the five scenarios; without this baseline
there is no quantitative way to claim that the learned policy improves
on the rule-based controller, nor to attribute the improvement to
specific state features via SHAP.

NEW GYMNASIUM API (5-tuple)
---------------------------
The environment was rewritten to expose the standard Gymnasium API::

    obs, info              = env.reset(seed=...)
    obs, reward, terminated, truncated, info = env.step(action)
    while not (terminated or truncated):
        ...

This runner uses that API directly and no longer carries compatibility
shims for the legacy 4-tuple / single-return paths.

OUTPUTS
-------
* ``trajectory_scenario{S}_ep{E}.csv`` -- one per (scenario, episode)
  with the per-step columns described in :data:`TRAJECTORY_COLUMNS`.
* ``summary.csv`` -- one row per (scenario, episode) with the aggregate
  metrics described in :data:`SUMMARY_COLUMNS`.

USAGE
-----
Run from the Member 1 root directory::

    python src/run_simulation.py --episodes 5 --seed 42
    python src/run_simulation.py --episodes 10 --seed 7 --scenarios 1,3,5
    python src/run_simulation.py --output-dir results
==============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  Make sibling modules (environment.py, baseline.py) importable regardless   #
#  of the caller's working directory.                                         #
# --------------------------------------------------------------------------- #
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from environment import NeonatalIncubatorEnv, Q_HEATER_MAX  # noqa: E402
from baseline import RuleBasedBaselineController  # noqa: E402


# --------------------------------------------------------------------------- #
#  Constants -- mirror of environment.py.  Kept locally so the reward-        #
#  recompute fallback does not depend on the env's private symbols.            #
# --------------------------------------------------------------------------- #
TARGET_TC: float = 37.0                  # core-temperature setpoint (deg C)
SPO2_TARGET_LOW: float = 91.0            # NICE preterm post-stab target band
SPO2_TARGET_HIGH: float = 95.0

# Hard safety thresholds (must match NeonatalIncubatorEnv).
TC_SAFE_LOW: float = 35.5
TC_SAFE_HIGH: float = 38.0
SPO2_SAFE_LOW: float = 85.0
SPO2_SAFE_HIGH: float = 96.0

# FiO2 fraction range used to normalize the FiO2 control effort.
FIO2_MIN: float = 0.21
FIO2_MAX: float = 0.60

# Reward weights used by the recompute fallback (match env defaults).
W_THERMAL: float = 1.0
W_OXYGEN: float = 0.5
W_ENERGY: float = 0.05
W_SAFETY: float = 1.0
R_SAFETY_PENALTY: float = 100.0

# Number of FiO2 levels per heater level (Discrete(25) = 5 x 5).
N_O2_LEVELS: int = 5

# Observation vector indices for the 11-dim Box state (fallback path on
# KeyError -- the env's info dict already exposes all keys, so this is
# only used as a defensive fallback).
OBS_INDEX: Dict[str, int] = {
    "Tc": 0, "Tsk": 1, "Ta": 2, "C_O2": 3, "SpO2": 4, "Tamb": 5,
    "dTa_dt": 6, "dC_O2_dt": 7, "phase": 8, "disturbance": 9,
    "scenario_id": 10,
}

# Per-step trajectory columns (frozen contract for Members 2 & 3).
# Note: ``scenario_id`` is the 11th obs feature (index 10) added in v3.0
# (V2-04) to restore a fully-observable MDP.
TRAJECTORY_COLUMNS: List[str] = [
    "step", "action", "heater_idx", "o2_idx",
    "u_heater", "fio2_cmd", "Q_heater_W",
    "Tc", "Tsk", "Ta", "C_O2", "SpO2", "Tamb",
    "dTa_dt", "dC_O2_dt", "phase", "disturbance", "scenario_id",
    "reward", "r_thermal", "r_oxygen", "r_energy", "r_safety",
    "terminated", "truncated",
]

# Per-episode summary columns (frozen contract for Members 2 & 3).
SUMMARY_COLUMNS: List[str] = [
    "scenario", "episode", "total_reward",
    "mean_Tc", "std_Tc", "mean_SpO2", "std_SpO2",
    "mean_Tc_error", "mean_SpO2_error",
    "num_safety_violations", "control_effort",
    "final_Tc", "final_SpO2",
    "episode_length", "terminated_reason",
    # V4-03: episode-termination diagnostic
    "termination_step", "termination_action", "termination_cause", "termination_state",
]


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def decode_action(
    action: int, env: NeonatalIncubatorEnv
) -> Tuple[int, int, float, float, float]:
    """Decode a ``Discrete(25)`` action index into the controller tuple.

    Mirrors ``NeonatalIncubatorEnv.step``::

        heater_idx = action // env.n_o2_levels
        o2_idx     = action %  env.n_o2_levels
        u_heater   = env.heater_levels[heater_idx]   # fraction in [0, 1]
        fio2_cmd   = env.fio2_levels[o2_idx]         # fraction in [0.21, 0.60]
        Q_heater_W = u_heater * env.q_heater_max      # watts

    Parameters
    ----------
    action : int
        Discrete action in ``[0, 25)``.
    env : NeonatalIncubatorEnv
        Environment instance (provides ``heater_levels`` and
        ``fio2_levels``).

    Returns
    -------
    (heater_idx, o2_idx, u_heater, fio2_cmd, Q_heater_W) : tuple of
        (int, int, float, float, float).
    """
    heater_idx = int(action) // int(env.n_o2_levels)
    o2_idx = int(action) % int(env.n_o2_levels)
    u_heater = float(env.heater_levels[heater_idx])
    fio2_cmd = float(env.fio2_levels[o2_idx])
    q_heater_w = u_heater * float(env.q_heater_max)
    return heater_idx, o2_idx, u_heater, fio2_cmd, q_heater_w


def read_state_value(
    info: Dict[str, Any], obs: np.ndarray, key: str
) -> float:
    """Read a numeric state field from ``info`` (preferred), falling back
    to positional indexing into ``obs`` on ``KeyError``.

    The env's info dict exposes every state key, so the fallback path is
    defensive only.
    """
    try:
        return float(info[key])
    except (KeyError, TypeError):
        idx = OBS_INDEX.get(key)
        if idx is None or idx >= len(obs):
            return float("nan")
        return float(obs[idx])


def read_reward_components(
    info: Dict[str, Any],
    obs: np.ndarray,
    u_heater: float,
    fio2_cmd: float,
) -> Tuple[float, float, float, float]:
    """Return ``(r_thermal, r_oxygen, r_energy, r_safety)``.

    Reads from ``info`` when the keys are present; on ``KeyError``
    recomputes the components from the post-step state and the applied
    controls using the same formulas as ``NeonatalIncubatorEnv.step``.
    """
    info = info or {}
    try:
        r_thermal = float(info["r_thermal"])
        r_oxygen = float(info["r_oxygen"])
        r_energy = float(info["r_energy"])
        r_safety = float(info["r_safety"])
    except KeyError:
        tc = read_state_value(info, obs, "Tc")
        spo2 = read_state_value(info, obs, "SpO2")
        r_thermal = -W_THERMAL * ((tc - TARGET_TC) ** 2)
        if spo2 < SPO2_TARGET_LOW:
            spo2_err = (SPO2_TARGET_LOW - spo2) ** 2
        elif spo2 > SPO2_TARGET_HIGH:
            spo2_err = (spo2 - SPO2_TARGET_HIGH) ** 2
        else:
            spo2_err = 0.0
        r_oxygen = -W_OXYGEN * spo2_err
        u_fio2_norm = (fio2_cmd - FIO2_MIN) / (FIO2_MAX - FIO2_MIN)
        r_energy = -W_ENERGY * (u_heater ** 2 + u_fio2_norm ** 2)
        unsafe = (
            tc < TC_SAFE_LOW or tc > TC_SAFE_HIGH
            or spo2 < SPO2_SAFE_LOW or spo2 > SPO2_SAFE_HIGH
        )
        r_safety = -W_SAFETY * R_SAFETY_PENALTY if unsafe else 0.0
    return r_thermal, r_oxygen, r_energy, r_safety


def is_unsafe(info: Dict[str, Any], obs: np.ndarray) -> bool:
    """Return True if the state breaches any hard safety limit."""
    info = info or {}
    try:
        return not bool(info["safe"])
    except (KeyError, TypeError):
        tc = read_state_value(info, obs, "Tc")
        spo2 = read_state_value(info, obs, "SpO2")
        return (
            tc < TC_SAFE_LOW or tc > TC_SAFE_HIGH
            or spo2 < SPO2_SAFE_LOW or spo2 > SPO2_SAFE_HIGH
        )


def build_trajectory_row(
    step: int,
    action: int,
    heater_idx: int,
    o2_idx: int,
    u_heater: float,
    fio2_cmd: float,
    q_heater_w: float,
    obs: np.ndarray,
    info: Dict[str, Any],
    reward: float,
    r_thermal: float,
    r_oxygen: float,
    r_energy: float,
    r_safety: float,
    terminated: bool,
    truncated: bool,
) -> Dict[str, Any]:
    """Assemble a single trajectory CSV row, reading state values from
    ``info`` (preferred) with ``obs`` as the fallback on ``KeyError``."""
    return {
        "step": int(step),
        "action": int(action),
        "heater_idx": int(heater_idx),
        "o2_idx": int(o2_idx),
        "u_heater": float(u_heater),
        "fio2_cmd": float(fio2_cmd),
        "Q_heater_W": float(q_heater_w),
        "Tc": read_state_value(info, obs, "Tc"),
        "Tsk": read_state_value(info, obs, "Tsk"),
        "Ta": read_state_value(info, obs, "Ta"),
        "C_O2": read_state_value(info, obs, "C_O2"),
        "SpO2": read_state_value(info, obs, "SpO2"),
        "Tamb": read_state_value(info, obs, "Tamb"),
        "dTa_dt": read_state_value(info, obs, "dTa_dt"),
        "dC_O2_dt": read_state_value(info, obs, "dC_O2_dt"),
        "phase": read_state_value(info, obs, "phase"),
        "disturbance": read_state_value(info, obs, "disturbance"),
        "scenario_id": read_state_value(info, obs, "scenario_id"),
        "reward": float(reward),
        "r_thermal": float(r_thermal),
        "r_oxygen": float(r_oxygen),
        "r_energy": float(r_energy),
        "r_safety": float(r_safety),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def spo2_band_error(spo2: float) -> float:
    """Distance of a single SpO2 value from the [91, 95] target band."""
    if spo2 < SPO2_TARGET_LOW:
        return SPO2_TARGET_LOW - spo2
    if spo2 > SPO2_TARGET_HIGH:
        return spo2 - SPO2_TARGET_HIGH
    return 0.0


def terminated_reason_label(
    terminated: bool, truncated: bool
) -> str:
    """Human-readable reason the episode ended."""
    if terminated:
        return "safety_violation"
    if truncated:
        return "max_steps_reached"
    return "in_progress"


# --------------------------------------------------------------------------- #
#  Episode runner                                                             #
# --------------------------------------------------------------------------- #
def run_episode(
    env: NeonatalIncubatorEnv,
    controller: RuleBasedBaselineController,
    scenario: int,
    episode: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run a single evaluation episode using the new Gymnasium API.

    Parameters
    ----------
    env : NeonatalIncubatorEnv
        Environment instance (will be ``reset(seed=...)`` internally).
    controller : RuleBasedBaselineController
        Baseline controller whose ``select_action(state) -> int`` drives
        the episode.
    scenario : int
        Scenario id (used for per-(scenario, episode) seeding and for
        the summary row).
    episode : int
        Episode index (0-based; used for seeding).
    seed : int
        Base RNG seed.

    Returns
    -------
    trajectory_df : pandas.DataFrame
        Per-step trajectory with columns :data:`TRAJECTORY_COLUMNS`.
    summary_row : dict
        Per-episode aggregate metrics (one row of the summary table).
    """
    # Per-(scenario, episode) reproducible seeding.  Belt-and-suspenders
    # for envs that rely on numpy's global RNG; the reference env uses
    # its own ``self._np_random`` seeded via the constructor.
    episode_seed = int(seed) + int(scenario) * 1000 + int(episode)
    np.random.seed(episode_seed)

    controller.reset()
    obs, info = env.reset(seed=episode_seed)

    rows: List[Dict[str, Any]] = []

    # ---- Step 0: log the initial (post-reset) state ------------------- #
    # No action has been taken yet, so action-decoded fields are NaN / -1.
    rows.append(build_trajectory_row(
        step=0, action=-1, heater_idx=-1, o2_idx=-1,
        u_heater=float("nan"), fio2_cmd=float("nan"),
        q_heater_w=float("nan"),
        obs=obs, info=info, reward=0.0,
        r_thermal=0.0, r_oxygen=0.0, r_energy=0.0, r_safety=0.0,
        terminated=False, truncated=False,
    ))

    step = 0
    total_reward = 0.0
    control_effort_accum = 0.0
    n_actions = 0
    safety_violations = 0
    terminated = False
    truncated = False

    # ---- Roll out the episode using the new Gymnasium API ------------- #
    while not (terminated or truncated):
        action = int(controller.select_action(obs))
        heater_idx, o2_idx, u_heater, fio2_cmd, q_heater_w = decode_action(
            action, env
        )

        obs, reward, terminated, truncated, info = env.step(action)

        r_thermal, r_oxygen, r_energy, r_safety = read_reward_components(
            info, obs, u_heater, fio2_cmd
        )

        step += 1
        rows.append(build_trajectory_row(
            step=step, action=action, heater_idx=heater_idx,
            o2_idx=o2_idx, u_heater=u_heater, fio2_cmd=fio2_cmd,
            q_heater_w=q_heater_w,
            obs=obs, info=info, reward=float(reward),
            r_thermal=r_thermal, r_oxygen=r_oxygen,
            r_energy=r_energy, r_safety=r_safety,
            terminated=terminated, truncated=truncated,
        ))

        total_reward += float(reward)
        # Control effort = mean(u_heater + u_fio2_norm); accumulate the
        # per-step sum and divide by the number of actions below.
        u_fio2_norm = (fio2_cmd - env.fio2_min) / (env.fio2_max - env.fio2_min)
        control_effort_accum += (u_heater + u_fio2_norm)
        n_actions += 1
        if is_unsafe(info, obs):
            safety_violations += 1

    traj_df = pd.DataFrame(rows, columns=TRAJECTORY_COLUMNS)

    # ---- Aggregate summary metrics ------------------------------------ #
    # State statistics span the full trajectory (initial state included,
    # since it is a real state of the system the controller must recover
    # from -- especially relevant for Scenario 4).
    tc_series = traj_df["Tc"].astype(float)
    spo2_series = traj_df["SpO2"].astype(float)

    spo2_err_series = spo2_series.apply(spo2_band_error)
    reason = terminated_reason_label(terminated, truncated)

    # V4-03: episode-termination diagnostic (cause, step, action, state at
    # termination). Captures WHY the episode ended, useful for training-risk
    # analysis (e.g. early termination under random exploration).
    last_row = traj_df.iloc[-1]
    if terminated:
        term_step = int(last_row["step"])
        term_action = int(last_row["action"])
        term_state = (f"Tc={last_row['Tc']:.2f},SpO2={last_row['SpO2']:.2f}")
        # Determine specific cause
        if last_row["Tc"] < 35.5:
            term_cause = "hypothermia"
        elif last_row["Tc"] > 38.0:
            term_cause = "hyperthermia"
        elif last_row["SpO2"] < 85.0:
            term_cause = "hypoxia"
        elif last_row["SpO2"] > 96.0:
            term_cause = "hyperoxia"
        else:
            term_cause = "safety_breach"
    elif truncated:
        term_step = int(last_row["step"])
        term_action = int(last_row["action"])
        term_state = (f"Tc={last_row['Tc']:.2f},SpO2={last_row['SpO2']:.2f}")
        term_cause = "max_steps"
    else:
        term_step = 0
        term_action = -1
        term_state = ""
        term_cause = "unknown"

    summary_row: Dict[str, Any] = {
        "scenario": int(scenario),
        "episode": int(episode),
        "total_reward": float(total_reward),
        "mean_Tc": float(tc_series.mean()),
        "std_Tc": float(tc_series.std(ddof=0)),
        "mean_SpO2": float(spo2_series.mean()),
        "std_SpO2": float(spo2_series.std(ddof=0)),
        "mean_Tc_error": float((tc_series - TARGET_TC).abs().mean()),
        "mean_SpO2_error": float(spo2_err_series.mean()),
        "num_safety_violations": int(safety_violations),
        "control_effort": float(control_effort_accum / max(n_actions, 1)),
        "final_Tc": float(traj_df["Tc"].iloc[-1]),
        "final_SpO2": float(traj_df["SpO2"].iloc[-1]),
        "episode_length": int(n_actions),
        "terminated_reason": reason,
        # V4-03: detailed termination diagnostic
        "termination_step": term_step,
        "termination_action": term_action,
        "termination_cause": term_cause,
        "termination_state": term_state,
    }
    return traj_df, summary_row


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_simulation.py",
        description=(
            "Run the rule-based baseline controller across incubator "
            "scenarios and persist per-episode trajectories + a summary "
            "table. Member 1 deliverable: produces the baseline numbers "
            "that the DDQN agent (Member 2) must beat."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run per scenario (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Base RNG seed. Per-(scenario, episode) seed is derived as "
            "seed + scenario*1000 + episode for reproducibility "
            "(default: 42)."
        ),
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated scenario ids to run (default: 1,2,3,4,5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help=(
            "Directory to write trajectory CSVs and summary.csv "
            "(default: results). Created if it does not exist. "
            "Interpreted relative to the caller's working directory, so "
            "running ``python src/run_simulation.py`` from the member_1 "
            "root writes to ``member_1/results/``."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional YAML config path (default: ../config.yaml). Applied "
            "to BOTH the environment and the baseline (custom action grids "
            "are honoured via RuleBasedBaselineController.from_env)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: run all (scenario, episode) combos and persist results."""
    args = parse_args(argv)

    try:
        scenario_ids = [
            int(s.strip()) for s in args.scenarios.split(",") if s.strip()
        ]
    except ValueError as exc:
        print(
            f"[run_simulation] could not parse --scenarios "
            f"{args.scenarios!r}: {exc}",
            file=sys.stderr,
        )
        return 2
    if not scenario_ids:
        print(
            "[run_simulation] no scenarios parsed from --scenarios; "
            "aborting.",
            file=sys.stderr,
        )
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Member 1 -- Baseline Simulation Runner")
    print("Neonatal Incubator Temperature & Oxygen Regulation (DDQN + SHAP)")
    print("=" * 72)
    print(f"Scenarios    : {scenario_ids}")
    print(f"Episodes/scn : {args.episodes}")
    print(f"Base seed    : {args.seed}")
    print(f"Output dir   : {output_dir.resolve()}")
    print("-" * 72)

    summary_rows: List[Dict[str, Any]] = []
    total_traj_files = 0

    for scenario in scenario_ids:
        for episode in range(args.episodes):
            # Re-instantiate the env per (scenario, episode) so the env's
            # private RNG is re-seeded -> fully reproducible episodes.
            ep_seed = args.seed + scenario * 1000 + episode
            env = NeonatalIncubatorEnv(scenario=scenario, seed=ep_seed,
                                       config_path=args.config)
            # Fix B: baseline grid always matches the env's actual grid.
            controller = RuleBasedBaselineController.from_env(
                env, config_path=args.config)
            try:
                traj_df, summary_row = run_episode(
                    env, controller, scenario, episode, args.seed
                )
            finally:
                if hasattr(env, "close"):
                    try:
                        env.close()
                    except Exception:  # pragma: no cover - defensive
                        pass

            traj_path = output_dir / (
                f"trajectory_scenario{scenario}_ep{episode}.csv"
            )
            traj_df.to_csv(traj_path, index=False)
            total_traj_files += 1
            summary_rows.append(summary_row)
            print(
                f"  [scn {scenario} | ep {episode}] "
                f"total_reward={summary_row['total_reward']:10.2f}  "
                f"mean_Tc={summary_row['mean_Tc']:.3f}  "
                f"mean_SpO2={summary_row['mean_SpO2']:.2f}  "
                f"safety_viol={summary_row['num_safety_violations']:3d}  "
                f"ctrl_eff={summary_row['control_effort']:.3f}  "
                f"len={summary_row['episode_length']:3d}  "
                f"reason={summary_row['terminated_reason']}  "
                f"-> {traj_path.name}"
            )

    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("-" * 72)
    print("Per-episode summary:")
    print(summary_df.to_string(index=False))
    print("-" * 72)
    print(
        f"Wrote {total_traj_files} trajectory CSV(s) and summary.csv "
        f"to {output_dir.resolve()}"
    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
