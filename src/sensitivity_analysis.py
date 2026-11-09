#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis (V4-04 fix)
================================
Varies key physics parameters independently around their documented baseline
(-20%, nominal, +20%) and records how the rule-based baseline controller
performs under each perturbation.

Parameters swept: k_loss, c_air, c_core, k_spo2, spo2_gain
Scenarios tested: 1 (normal), 3 (porthole), 5 (oxygen disturbance)
Metrics per condition: mean_reward, mean_Tc_error, mean_SpO2_error,
                       num_safety_violations, episode_length

Outputs:
  results/sensitivity_results.csv  -- full per-condition metrics
  plots/sensitivity_reward.png     -- bar chart of reward by param/scenario
  plots/sensitivity_tc_error.png  -- bar chart of Tc error
  plots/sensitivity_spo2_error.png -- bar chart of SpO2 error

Usage:
  python src/sensitivity_analysis.py
"""
from __future__ import annotations
import os, sys, copy, itertools
import numpy as np
import pandas as pd

# Make src importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import NeonatalIncubatorEnv, load_config, SCENARIO_NAMES
from baseline import RuleBasedBaselineController
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False


# Parameters to sweep: (config_key, display_name, nominal_value)
BASE_CFG = load_config()
PARAMS = [
    ("k_loss", "K_LOSS", BASE_CFG["k_loss"]),
    ("c_air", "C_AIR", BASE_CFG["c_air"]),
    ("c_core", "C_CORE", BASE_CFG["c_core"]),
    ("k_spo2", "K_SPO2", BASE_CFG["k_spo2"]),
    ("spo2_gain", "SPO2_GAIN", BASE_CFG["spo2_gain"]),
]
SCENARIOS = [1, 3, 5]  # normal, porthole, oxygen disturbance
PERTURBATIONS = [-0.20, 0.0, 0.20]  # -20%, nominal, +20%
EPISODES_PER_COND = 3
BASE_SEED = 42


def run_episode(env: NeonatalIncubatorEnv, controller: RuleBasedBaselineController,
                seed: int) -> dict:
    """Run one episode with the baseline controller; return metrics."""
    obs, info = env.reset(seed=seed)
    controller.reset()
    total_reward = 0.0
    tcs, spo2s = [], []
    safety_violations = 0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(controller.select_action(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        tcs.append(float(info["Tc"]))
        spo2s.append(float(info["SpO2"]))
        if not info["safe"]:
            safety_violations += 1
        steps += 1
    return {
        "total_reward": total_reward,
        "mean_Tc": float(np.mean(tcs)) if tcs else 0.0,
        "mean_SpO2": float(np.mean(spo2s)) if spo2s else 0.0,
        "mean_Tc_error": float(np.mean([abs(t - 37.0) for t in tcs])) if tcs else 0.0,
        "mean_SpO2_error": float(np.mean([abs(s - 91.0) if s < 91 else
                                          (abs(s - 95.0) if s > 95 else 0.0)
                                          for s in spo2s])) if spo2s else 0.0,
        "num_safety_violations": safety_violations,
        "episode_length": steps,
    }


def make_env_with_override(param_key: str, value: float, scenario: int) -> tuple:
    """Build an env + controller with one parameter overridden via temp YAML."""
    import tempfile, yaml
    cfg = {
        "physics": {},
    }
    # Place the override in the right YAML section.
    if param_key in ("k_loss", "k_baby", "c_air", "c_core",
                     "porthole_h_c_factor"):
        cfg["physics"] = {"thermal": {param_key: value}}
    elif param_key in ("k_blend", "k_leak", "desat_offset"):
        cfg["physics"] = {"oxygen": {param_key: value}}
    elif param_key in ("k_spo2", "spo2_base", "spo2_gain", "spo2_min", "spo2_max"):
        cfg["physics"] = {"spo2": {param_key: value}}
    else:
        cfg = {"environment": {param_key: value}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        path = f.name
    env = NeonatalIncubatorEnv(scenario=scenario, config_path=path)
    ctrl = RuleBasedBaselineController.from_env(env, config_path=path)
    os.unlink(path)
    return env, ctrl


def main():
    print("=" * 70)
    print("V4-04 Sensitivity Analysis")
    print(f"Parameters: {[p[1] for p in PARAMS]}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Perturbations: {PERTURBATIONS}  ({EPISODES_PER_COND} episodes each)")
    print("=" * 70)
    rows = []
    for param_key, display, nominal in PARAMS:
        for pert in PERTURBATIONS:
            value = nominal * (1.0 + pert)
            if value <= 0:
                # Skip non-physical (e.g. k_loss=0 would be caught by validation).
                for scn in SCENARIOS:
                    rows.append({
                        "parameter": display, "perturbation": pert,
                        "value": value, "scenario": scn,
                        "total_reward": float("nan"),
                        "mean_Tc_error": float("nan"),
                        "mean_SpO2_error": float("nan"),
                        "num_safety_violations": -1,
                        "episode_length": 0,
                        "note": "skipped (non-physical)",
                    })
                continue
            for scn in SCENARIOS:
                rewards, tc_errs, spo2_errs, viols, lens = [], [], [], [], []
                for ep in range(EPISODES_PER_COND):
                    seed = BASE_SEED + scn * 1000 + ep
                    try:
                        env, ctrl = make_env_with_override(param_key, value, scn)
                        m = run_episode(env, ctrl, seed)
                        rewards.append(m["total_reward"])
                        tc_errs.append(m["mean_Tc_error"])
                        spo2_errs.append(m["mean_SpO2_error"])
                        viols.append(m["num_safety_violations"])
                        lens.append(m["episode_length"])
                        note = ""
                    except (ValueError, ZeroDivisionError) as e:
                        rewards.append(float("nan"))
                        tc_errs.append(float("nan"))
                        spo2_errs.append(float("nan"))
                        viols.append(-1)
                        lens.append(0)
                        note = f"error: {type(e).__name__}"
                rows.append({
                    "parameter": display,
                    "perturbation": pert,
                    "value": round(value, 4),
                    "scenario": scn,
                    "total_reward": float(np.nanmean(rewards)),
                    "mean_Tc_error": float(np.nanmean(tc_errs)),
                    "mean_SpO2_error": float(np.nanmean(spo2_errs)),
                    "num_safety_violations": float(np.nanmean(viols)),
                    "episode_length": float(np.nanmean(lens)),
                    "note": note,
                })
                r = rows[-1]
                print(f"  {display:10s} pert={pert:+.0%} val={r['value']:8.2f}  "
                      f"scn={scn}  reward={r['total_reward']:8.2f}  "
                      f"Tc_err={r['mean_Tc_error']:.3f}  "
                      f"SpO2_err={r['mean_SpO2_error']:.3f}  "
                      f"viol={r['num_safety_violations']:.0f}  "
                      f"len={r['episode_length']:.0f}")
    df = pd.DataFrame(rows)
    # V4-fix: output relative to the member_1 root (parent of src/), regardless
    # of the caller's CWD, so results always land in member_1/results/.
    _member_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(_member_root, "results")
    plots_dir = os.path.join(_member_root, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "sensitivity_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} rows)")

    # Generate plots.
    valid = df.dropna(subset=["total_reward"])
    for metric, fname, title in [
        ("total_reward", "sensitivity_reward.png", "Mean Total Reward"),
        ("mean_Tc_error", "sensitivity_tc_error.png", "Mean |Tc - 37| (°C)"),
        ("mean_SpO2_error", "sensitivity_spo2_error.png", "Mean SpO2 Band Error"),
    ]:
        fig, axes = plt.subplots(1, len(PARAMS), figsize=(4 * len(PARAMS), 4),
                                sharey=True)
        for ax, (pk, disp, _) in zip(axes, PARAMS):
            sub = valid[valid["parameter"] == disp]
            piv = sub.pivot_table(index="perturbation", columns="scenario",
                                  values=metric, aggfunc="mean")
            piv.plot(kind="bar", ax=ax, legend=False)
            ax.set_title(disp)
            ax.set_xlabel("Perturbation")
            ax.axvline(x=1, color="gray", linestyle="--", alpha=0.3)
        axes[-1].legend(title="Scenario", loc="best")
        fig.suptitle(f"Sensitivity: {title}")
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(plots_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  Wrote {os.path.join(plots_dir, fname)}")
    print("\nSensitivity analysis complete.")


if __name__ == "__main__":
    main()
