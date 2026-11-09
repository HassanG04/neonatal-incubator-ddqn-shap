#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validation.py -- Multi-Scenario Decision Validation and Baseline Benchmarking
=============================================================================
Project   : Project 04 - Member 03 (Explainability, Validation & Integration)
Topic     : Neonatal Incubator Temperature and Oxygen Regulation (DDQN + SHAP)
Reference : Member 1 Environment (v3.1) & Baseline (PID + bang-bang)
            Member 2 DDQN Agent & Evaluation Protocol

Audit Corrections & Improvements:
---------------------------------
1. Strict State-Action Alignment:
   Explicitly records:
     - decision_state: 11 observation features before action selection.
     - selected_action: discrete action chosen from decision_state.
     - decision_q_values: Q-values evaluated at decision_state (including all 25 Q-values).
     - shap_values: feature attributions evaluated at decision_state for selected_action.
     - reward: immediate transition reward resulting from the action.
     - next_state: 11 observation features resulting after action execution.
2. Stratified on-manifold background sampling across complete 200-step simulation timeline.
3. Precise reporting of modeled hard-threshold termination avoidance (no false "clinical survival" claims).
4. Rigorous action-switching and chattering analysis.
5. Head-to-head benchmarking against baseline controller across 5 seeds x 5 scenarios (25 DDQN + 25 Baseline episodes).

>>> SIMULATION-ONLY RESEARCH PROJECT - NOT FOR CLINICAL USE <<<
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Suppress TF verbose logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_CUR_DIR, "src")
for p in (_CUR_DIR, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

from baseline import RuleBasedBaselineController
from ddqn_agent import DDQNAgent, select_device
from environment import (
    ACTION_TABLE,
    NeonatalIncubatorEnv,
    OBS_NAMES,
    SCENARIO_NAMES,
)
from rl_utils import (
    AgentPolicy,
    BaselinePolicy,
    Policy,
    episode_metrics,
    set_global_seeds,
    test_seed,
)
from shap_explainer import (
    DDQNShapExplainer,
    prepare_explanation,
    load_trained_agent,
)

logger = logging.getLogger("validation")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def evaluate_trajectory(
    env: NeonatalIncubatorEnv,
    policy: Policy,
    seed: int,
    explainer: Optional[DDQNShapExplainer] = None,
    explain_steps: Optional[Sequence[int]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute a single episode with rigorous state-action alignment.

    For every transition t:
      1. Decision state s_t is observed.
      2. Policy selects action a_t based on s_t.
      3. Q-values and SHAP values are evaluated at s_t for a_t.
      4. Environment executes a_t, returning r_{t+1}, s_{t+1}, and flags.
      5. The complete aligned transition (s_t, a_t, Q(s_t), phi(s_t, a_t), r_{t+1}, s_{t+1}) is stored.

    Parameters
    ----------
    env : NeonatalIncubatorEnv
    policy : Policy
    seed : int
    explainer : Optional[DDQNShapExplainer]
    explain_steps : Optional[Sequence[int]]

    Returns
    -------
    trajectory_rows : List[Dict[str, Any]]
    metrics : Dict[str, Any]
    """
    obs, info = env.reset(seed=int(seed))
    policy.reset(env)

    explain_set = None if explain_steps is None else set(explain_steps)
    rows: List[Dict[str, Any]] = []
    terminated = truncated = False
    step = 0

    # Initial state row (before any action is taken; step 0 anchor for episode_metrics)
    initial_row = _format_initial_record(obs=obs, info=info, env=env, policy=policy)
    rows.append(initial_row)

    while not (terminated or truncated):
        # 1. Decision State: observation BEFORE taking action
        decision_state = obs.copy()

        # 2. Selected Action: chosen based on decision_state
        action = int(policy.act(decision_state))

        # 3. Evaluate Q-values at decision_state
        decision_q_values: Optional[np.ndarray] = None
        if isinstance(policy, AgentPolicy):
            norm_s = policy.agent.preprocess(decision_state)
            decision_q_values = policy.agent.q_values(norm_s)[0]

        # 4. Evaluate SHAP attributions at decision_state if requested
        shap_info: Dict[str, Any] = {}
        if explainer is not None and decision_q_values is not None and (explain_set is None or step in explain_set):
            base, phi, residual = explainer.decision_values(norm_s, action, decision_q_values[action])
            shap_info = {"shap_base_val": base, "shap_efficiency_error": residual}
            for rank, idx in enumerate(np.argsort(np.abs(phi))[::-1][:3], start=1):
                shap_info[f"shap_top{rank}_feature"] = OBS_NAMES[idx]
                shap_info[f"shap_top{rank}_val"] = float(phi[idx])
            shap_info.update({f"shap_{name}": float(value) for name, value in zip(OBS_NAMES, phi)})

        # 5. Environment step execution: yields next state, reward, and flags
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        step += 1

        # 6. Format aligned record containing both decision_state and next_state
        row = _format_transition_record(
            step=step,
            decision_state=decision_state,
            selected_action=action,
            decision_q_values=decision_q_values,
            reward=float(reward),
            next_state=next_obs,
            terminated=bool(terminated),
            truncated=bool(truncated),
            step_info=step_info,
            env=env,
            policy=policy,
            shap_info=shap_info,
        )
        rows.append(row)
        obs = next_obs

    for row in rows:
        row["seed"] = int(seed)

    # Calculate summary metrics using rl_utils
    metrics = episode_metrics(rows, env)
    metrics["seed"] = int(seed)
    metrics["scenario"] = int(env.scenario)
    metrics["scenario_name"] = SCENARIO_NAMES[env.scenario]
    metrics["policy"] = policy.name

    # Additional action-switching and minimum SpO2 diagnostics
    actions_taken = [r["selected_action"] for r in rows[1:]]
    if len(actions_taken) > 1:
        switches = int(sum(1 for i in range(1, len(actions_taken)) if actions_taken[i] != actions_taken[i - 1]))
    else:
        switches = 0
    metrics["action_switches"] = switches
    metrics["action_switch_ratio"] = float(switches / max(1, len(actions_taken) - 1))

    spo2_vals = [r["SpO2"] for r in rows[1:]]
    metrics["spo2_min"] = float(min(spo2_vals)) if spo2_vals else np.nan
    metrics["spo2_max"] = float(max(spo2_vals)) if spo2_vals else np.nan

    return rows, metrics


def _format_initial_record(
    obs: np.ndarray,
    info: Dict[str, Any],
    env: NeonatalIncubatorEnv,
    policy: Policy,
) -> Dict[str, Any]:
    """Format step 0 pre-action anchor record."""
    rec: Dict[str, Any] = {
        "step": 0,
        "time_sec": 0.0,
        "selected_action": -1,
        "action": -1,  # Compatibility alias
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "policy": policy.name,
        "scenario": env.scenario,
        "scenario_name": SCENARIO_NAMES[env.scenario],
        "heater_idx": -1,
        "o2_idx": -1,
        "heater_level": np.nan,
        "u_heater": np.nan,
        "Q_heater_W": np.nan,
        "fio2_cmd": np.nan,
        "fio2_command": np.nan,
        "decision_q_chosen": np.nan,
        "q_chosen": np.nan,
        "decision_q_max": np.nan,
        "decision_q_mean": np.nan,
        "decision_q_min": np.nan,
    }

    # Populate both decision_state and next_state as the initial observation
    for idx, name in enumerate(OBS_NAMES):
        rec[f"decision_obs_{name}"] = float(obs[idx])
        rec[f"next_obs_{name}"] = float(obs[idx])

    # Populate post-transition physical states for metrics compatibility
    rec["Tc"] = float(obs[0])
    rec["Tsk"] = float(obs[1])
    rec["Ta"] = float(obs[2])
    rec["C_O2"] = float(obs[3])
    rec["SpO2"] = float(obs[4])
    rec["Tamb"] = float(obs[5])
    rec["tc_err"] = abs(float(obs[0]) - env.tc_setpoint)
    rec["tc_in_comfort"] = 36.5 <= float(obs[0]) <= 37.5
    rec["tc_in_safe"] = env.tc_safe_low <= float(obs[0]) <= env.tc_safe_high
    rec["spo2_in_comfort"] = env.spo2_target_low <= float(obs[4]) <= env.spo2_target_high
    rec["spo2_in_safe"] = env.spo2_safe_low <= float(obs[4]) <= env.spo2_safe_high

    return rec


def _format_transition_record(
    step: int,
    decision_state: np.ndarray,
    selected_action: int,
    decision_q_values: Optional[np.ndarray],
    reward: float,
    next_state: np.ndarray,
    terminated: bool,
    truncated: bool,
    step_info: Dict[str, Any],
    env: NeonatalIncubatorEnv,
    policy: Policy,
    shap_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Format an aligned transition: decision_state -> selected_action -> reward, next_state."""
    rec: Dict[str, Any] = {
        "step": step,
        "time_sec": step * env.dt,
        "selected_action": selected_action,
        "action": selected_action,  # Compatibility alias for episode_metrics
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "policy": policy.name,
        "scenario": env.scenario,
        "scenario_name": SCENARIO_NAMES[env.scenario],
    }

    # 1. Store all 11 decision-state features (state BEFORE action)
    for idx, name in enumerate(OBS_NAMES):
        rec[f"decision_obs_{name}"] = float(decision_state[idx])

    # 2. Store actuator parameters for selected_action
    if 0 <= selected_action < len(env.action_table):
        act_meta = env.action_table[selected_action]
        rec["heater_idx"] = act_meta["heater_idx"]
        rec["o2_idx"] = act_meta["o2_idx"]
        rec["heater_level"] = act_meta["heater_level"]
        rec["u_heater"] = float(step_info.get("u_heater", act_meta["heater_level"]))
        rec["Q_heater_W"] = float(step_info.get("Q_heater_W", act_meta["Q_heater_W"]))
        rec["fio2_cmd"] = float(step_info.get("fio2_cmd", act_meta["fio2_command"] * 100.0))
        rec["fio2_command"] = act_meta["fio2_command"]
    else:
        rec["heater_idx"] = -1
        rec["o2_idx"] = -1
        rec["heater_level"] = np.nan
        rec["u_heater"] = np.nan
        rec["Q_heater_W"] = np.nan
        rec["fio2_cmd"] = np.nan
        rec["fio2_command"] = np.nan

    # 3. Store Q-values evaluated at decision_state
    if decision_q_values is not None:
        rec["decision_q_chosen"] = float(decision_q_values[selected_action])
        rec["q_chosen"] = float(decision_q_values[selected_action])  # Alias
        rec["decision_q_max"] = float(np.max(decision_q_values))
        rec["decision_q_mean"] = float(np.mean(decision_q_values))
        rec["decision_q_min"] = float(np.min(decision_q_values))
        # Store all 25 individual Q-values for complete auditability
        for a_idx, q_val in enumerate(decision_q_values):
            rec[f"q_val_{a_idx}"] = float(q_val)
    else:
        rec["decision_q_chosen"] = np.nan
        rec["q_chosen"] = np.nan
        rec["decision_q_max"] = np.nan
        rec["decision_q_mean"] = np.nan
        rec["decision_q_min"] = np.nan

    # 4. Store SHAP explanation fields (evaluated at decision_state)
    if shap_info:
        rec.update(shap_info)

    # 5. Store all 11 next-state features (state AFTER action execution)
    for idx, name in enumerate(OBS_NAMES):
        rec[f"next_obs_{name}"] = float(next_state[idx])

    # 6. Post-transition physical states (used by episode_metrics)
    rec["Tc"] = float(next_state[0])
    rec["Tsk"] = float(next_state[1])
    rec["Ta"] = float(next_state[2])
    rec["C_O2"] = float(next_state[3])
    rec["SpO2"] = float(next_state[4])
    rec["Tamb"] = float(next_state[5])
    rec["tc_err"] = abs(float(next_state[0]) - env.tc_setpoint)
    rec["tc_in_comfort"] = 36.5 <= float(next_state[0]) <= 37.5
    rec["tc_in_safe"] = env.tc_safe_low <= float(next_state[0]) <= env.tc_safe_high
    rec["spo2_in_comfort"] = env.spo2_target_low <= float(next_state[4]) <= env.spo2_target_high
    rec["spo2_in_safe"] = env.spo2_safe_low <= float(next_state[4]) <= env.spo2_safe_high

    # 7. Environment reward breakdown and safety flags from step_info
    for k in ("r_thermal", "r_oxygen", "r_energy", "r_safety", "hypothermia", "hyperthermia", "hypoxia", "hyperoxia", "desat_offset"):
        if k in step_info:
            rec[k] = step_info[k]

    return rec


def summarize_episodes(frame):
    recovery = frame.recovery_steps.dropna()
    return {
        "mean_total_reward": float(frame.total_reward.mean()),
        "std_total_reward": float(frame.total_reward.std()) if len(frame) > 1 else None,
        "modeled_termination_avoidance_rate": float(frame.survived.mean()),
        "safety_threshold_violations": int(frame.safety_violation.sum()),
        "tc_mean_abs_err": float(frame.tc_mean_abs_err.mean()),
        "tc_time_in_comfort_band": float(frame.tc_time_in_band.mean()),
        "spo2_mean_band_err": float(frame.spo2_mean_band_err.mean()),
        "spo2_time_in_comfort_band": float(frame.spo2_time_in_band.mean()),
        "spo2_min_observed": float(frame.spo2_min.min()),
        "control_effort": float(frame.control_effort.mean()),
        "heater_energy_kJ": float(frame.heater_energy_kJ.mean()),
        "mean_action_switches": float(frame.action_switches.mean()),
        "mean_recovery_steps_observed": float(recovery.mean()) if len(recovery) else None,
        "episodes_with_observed_recovery": int(len(recovery)),
    }


def run_decision_validation(
    checkpoint_path: str = "models/ddqn_tuned_seed0/best.pt",
    output_dir: str = "outputs/member03/validation",
    episodes_per_scenario: int = 5,
    device: str = "cpu",
    seed_base: int = 42,
    shap_method: str = "deep",
    skip_shap: bool = False,
    context=None,
) -> Dict[str, Any]:
    """Execute complete validation across Normal, Abnormal Temp, and Abnormal O2 scenarios."""
    os.makedirs(output_dir, exist_ok=True)
    set_global_seeds(seed_base)

    logger.info(f"Loading DDQNAgent from '{checkpoint_path}'...")
    if episodes_per_scenario < 1:
        raise ValueError("episodes_per_scenario must be positive")
    if skip_shap:
        agent = load_trained_agent(checkpoint_path, device)
        explainer = None
    else:
        agent, _, _, _, explainer = context or prepare_explanation(checkpoint_path, device, seed_base, shap_method)
    ddqn_policy = AgentPolicy(agent, name="ddqn")
    baseline_policy = BaselinePolicy(name="baseline")

    validation_groups = {
        "normal": [1],
        "abnormal_temp": [2, 3, 4],
        "abnormal_o2": [5],
    }

    all_metrics: List[Dict[str, Any]] = []
    scenario_dfs: Dict[str, pd.DataFrame] = {}

    total_episodes_ddqn = 0
    total_episodes_base = 0

    for group_name, scenarios in validation_groups.items():
        logger.info(f"=== Processing Validation Group: {group_name.upper()} (Scenarios: {scenarios}) ===")
        group_rows: List[Dict[str, Any]] = []

        for scn in scenarios:
            scn_name = SCENARIO_NAMES[scn]
            logger.info(f"  -> Scenario {scn} ({scn_name}) across {episodes_per_scenario} held-out test seeds:")

            for ep_idx in range(episodes_per_scenario):
                ep_seed = test_seed(scn, ep_idx)

                # 1. Run DDQN Policy
                env_ddqn = NeonatalIncubatorEnv(scenario=scn)
                traj_ddqn, met_ddqn = evaluate_trajectory(
                    env=env_ddqn,
                    policy=ddqn_policy,
                    seed=ep_seed,
                    explainer=explainer,
                )
                met_ddqn["group"] = group_name
                met_ddqn["ep_index"] = ep_idx
                all_metrics.append(met_ddqn)
                group_rows.extend(traj_ddqn)
                total_episodes_ddqn += 1

                # 2. Run Baseline Policy (fair comparison on identical test seed)
                env_base = NeonatalIncubatorEnv(scenario=scn)
                traj_base, met_base = evaluate_trajectory(
                    env=env_base,
                    policy=baseline_policy,
                    seed=ep_seed,
                    explainer=None,
                )
                met_base["group"] = group_name
                met_base["ep_index"] = ep_idx
                all_metrics.append(met_base)
                group_rows.extend(traj_base)
                total_episodes_base += 1

                logger.info(
                    f"     Seed {ep_seed} [Ep {ep_idx}]: "
                    f"DDQN Ret={met_ddqn['total_reward']:.2f} (Tc Err={met_ddqn['tc_mean_abs_err']:.3f}C, Switches={met_ddqn['action_switches']}) vs "
                    f"Base Ret={met_base['total_reward']:.2f} (Tc Err={met_base['tc_mean_abs_err']:.3f}C, Switches={met_base['action_switches']})"
                )

        df_group = pd.DataFrame(group_rows)
        scenario_dfs[group_name] = df_group

    # Save scenario CSV files
    logger.info("Saving scenario trajectory CSV files...")
    normal_csv = os.path.join(output_dir, "normal_scenario_results.csv")
    temp_csv = os.path.join(output_dir, "abnormal_temperature_results.csv")
    o2_csv = os.path.join(output_dir, "abnormal_oxygen_results.csv")

    scenario_dfs["normal"].to_csv(normal_csv, index=False)
    scenario_dfs["abnormal_temp"].to_csv(temp_csv, index=False)
    scenario_dfs["abnormal_o2"].to_csv(o2_csv, index=False)

    # Save episode metrics CSV
    df_metrics = pd.DataFrame(all_metrics)
    metrics_csv = os.path.join(output_dir, "validation_episodes.csv")
    df_metrics.to_csv(metrics_csv, index=False)

    # Build comprehensive JSON summary with exact audit terminology
    summary_dict: Dict[str, Any] = {
        "checkpoint_path": checkpoint_path,
        "shap_configuration": {
            "skip_shap": skip_shap,
            "fallback_reason": explainer.fallback_reason if explainer else None,
            "shap_method_requested": shap_method,
            "shap_method_used": explainer.method if explainer is not None else "none",
        },
        "evaluation_design": {
            "episodes_per_scenario_per_controller": episodes_per_scenario,
            "total_ddqn_episodes": total_episodes_ddqn,
            "total_baseline_episodes": total_episodes_base,
            "total_evaluation_episodes": total_episodes_ddqn + total_episodes_base,
            "scenarios_evaluated": list(SCENARIO_NAMES.keys()),
        },
        "by_scenario": {},
        "overall": {},
    }

    for scn in sorted(SCENARIO_NAMES.keys()):
        scn_name = SCENARIO_NAMES[scn]
        df_scn = df_metrics[df_metrics["scenario"] == scn]

        ddqn_rows = df_scn[df_scn["policy"] == "ddqn"]
        base_rows = df_scn[df_scn["policy"] == "baseline"]

        summary_dict["by_scenario"][scn_name] = {
            "scenario_id": scn, "ddqn": summarize_episodes(ddqn_rows),
            "baseline": summarize_episodes(base_rows),
            "ddqn_advantage_return": float(ddqn_rows.total_reward.mean() - base_rows.total_reward.mean()),
        }

    ddqn = summarize_episodes(df_metrics[df_metrics.policy == "ddqn"])
    baseline = summarize_episodes(df_metrics[df_metrics.policy == "baseline"])
    summary_dict["overall"] = {"ddqn": ddqn, "baseline": baseline,
        "net_ddqn_improvement": ddqn["mean_total_reward"] - baseline["mean_total_reward"]}

    summary_json = os.path.join(output_dir, "validation_summary.json")
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary_dict, fh, indent=2, allow_nan=False)

    logger.info(f"Validation complete. Summary saved -> {summary_json}")
    logger.info(
        f"Net DDQN Return vs Baseline: "
        f"{summary_dict['overall']['ddqn']['mean_total_reward']:.2f} vs "
        f"{summary_dict['overall']['baseline']['mean_total_reward']:.2f} "
        f"(Net: {summary_dict['overall']['net_ddqn_improvement']:+.2f})"
    )
    return summary_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decision Validation Engine for Neonatal Incubator DDQN Agent"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/ddqn_tuned_seed0/best.pt",
        help="Path to trained DDQN checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/member03/validation",
        help="Output directory for validation CSVs and JSON summary",
    )
    parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        default=5,
        help="Number of held-out test episodes per scenario per controller",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device ('cpu' or 'gpu')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for evaluation",
    )
    parser.add_argument(
        "--shap-method",
        type=str,
        choices=["deep", "kernel"],
        default="deep",
        help="SHAP method for every DDQN decision",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP calculations entirely during validation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_decision_validation(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        episodes_per_scenario=args.episodes_per_scenario,
        device=args.device,
        seed_base=args.seed,
        shap_method=args.shap_method,
        skip_shap=args.skip_shap,
    )
