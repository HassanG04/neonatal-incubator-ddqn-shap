#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_controllers.py -- Fair held-out evaluation of Baseline, DDQN, and DQN.

Evaluates every controller on IDENTICAL held-out (scenario, test_seed) pairs
(``rl_utils.test_seed(scenario, ep_idx)``, disjoint from training and
validation seeds) using the exact same environment, step budget, and metric
definitions.

Outputs (under ``--output-dir``, default ``results/ddqn``)
----------------------------------------------------------
* ``eval_episodes.csv``         per-episode metrics for every controller/seed
* ``eval_by_scenario.csv``      per-scenario mean, std, 95% CI & seed stats
* ``eval_aggregate.csv``        aggregate metrics across all 5 scenarios
* ``eval_paired_diff.csv``      paired (DDQN - Baseline) & (DDQN - DQN) diffs
* ``eval_trajectories_rep.csv`` step-by-step trajectories on test ep 0
* ``eval_manifest.json``        full machine-readable run manifest

Plots (under ``--plots-dir``, default ``plots``)
------------------------------------------------
* ``ddqn_learning_curves.png``          training & validation curves across seeds
* ``ddqn_loss_and_q_diagnostics.png``   loss, Q-values, TD error & grad norms
* ``ddqn_vs_baseline_by_scenario.png``  per-scenario comparison bars (mean +/- std)
* ``ddqn_seed_variability.png``         variation across trained seeds
* ``ddqn_timeseries_scenarios.png``     Tc, SpO2, heater & FiO2 trajectories
* ``ddqn_action_timelines.png``         action timelines & actuator distributions

>>> SIMULATION-ONLY ACADEMIC PROJECT -- NOT FOR CLINICAL USE. <<<
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ddqn_agent import DDQNAgent, select_device  # noqa: E402
from environment import NeonatalIncubatorEnv, SCENARIO_NAMES  # noqa: E402
from rl_utils import (  # noqa: E402
    PROJECT_ROOT,
    AgentPolicy,
    BaselinePolicy,
    Policy,
    build_manifest,
    run_episode,
    set_global_seeds,
    test_seed,
    write_json,
)

METRIC_COLS: List[str] = [
    "total_reward",
    "episode_length",
    "survived",
    "safety_violation",
    "tc_mean_abs_err",
    "tc_max_abs_err",
    "tc_time_in_band",
    "spo2_mean_band_err",
    "spo2_max_band_err",
    "spo2_time_in_band",
    "mean_heater_frac",
    "mean_fio2_norm",
    "control_effort",
    "heater_energy_kJ",
    "action_switches",
    "recovery_steps",
    "overshoot",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models-dir", type=str, default=os.path.join(PROJECT_ROOT, "models"))
    p.add_argument("--train-results-dir", type=str,
                   default=os.path.join(PROJECT_ROOT, "results", "ddqn"))
    p.add_argument("--output-dir", type=str,
                   default=os.path.join(PROJECT_ROOT, "results", "ddqn"))
    p.add_argument("--plots-dir", type=str, default=os.path.join(PROJECT_ROOT, "plots"))
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--episodes-per-scenario", type=int, default=10,
                   help="Held-out test episodes per scenario (test_seed(scn, i))")
    p.add_argument("--ddqn-runs", nargs="*", default=None,
                   help="Explicit DDQN run names under models/ (default: auto-discover ddqn_seed*)")
    p.add_argument("--dqn-runs", nargs="*", default=None,
                   help="Explicit DQN run names under models/ (default: auto-discover dqn_seed*)")
    p.add_argument("--checkpoint", choices=["best", "final"], default="best")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _discover_runs(models_dir: str, prefix: str, ckpt_name: str) -> List[Tuple[str, str, int]]:
    """Return sorted list of ``(run_name, ckpt_path, seed)`` for ``{prefix}_seed*``."""
    out: List[Tuple[str, str, int]] = []
    if not os.path.isdir(models_dir):
        return out
    for name in sorted(os.listdir(models_dir)):
        if not name.startswith(f"{prefix}_seed") or name.endswith("_smoke"):
            continue
        ckpt = os.path.join(models_dir, name, f"{ckpt_name}.pt")
        if not os.path.isfile(ckpt):
            continue
        suffix = name[len(f"{prefix}_seed"):]
        try:
            seed = int(suffix.split("_")[0])
        except ValueError:
            seed = len(out)
        out.append((name, ckpt, seed))
    out.sort(key=lambda t: t[2])
    return out


def _resolve_runs(models_dir: str, explicit: Optional[List[str]],
                  prefix: str, ckpt_name: str) -> List[Tuple[str, str, int]]:
    if explicit is None:
        return _discover_runs(models_dir, prefix, ckpt_name)
    out: List[Tuple[str, str, int]] = []
    for idx, name in enumerate(explicit):
        ckpt = (name if os.path.isfile(name)
                else os.path.join(models_dir, name, f"{ckpt_name}.pt"))
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        seed = idx
        if "_seed" in name:
            try:
                seed = int(name.split("_seed", 1)[1].split("_")[0].split("/")[0])
            except ValueError:
                seed = idx
        out.append((os.path.basename(os.path.dirname(ckpt)) if os.path.isfile(name) else name,
                    ckpt, seed))
    return out


def _ci95(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))


def _summarise_group(df: pd.DataFrame) -> Dict[str, Any]:
    row: Dict[str, Any] = {"n_episodes": int(len(df))}
    for col in METRIC_COLS:
        vals = df[col].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        row[f"{col}_mean"] = float(np.mean(finite)) if finite.size else np.nan
        row[f"{col}_std"] = float(np.std(finite, ddof=0)) if finite.size else np.nan
        row[f"{col}_ci95"] = _ci95(finite) if finite.size else np.nan
    # If multiple model seeds are present, compute inter-seed std of seed means
    if "model_seed" in df.columns and df["model_seed"].nunique() > 1:
        seed_means = df.groupby("model_seed")["total_reward"].mean().to_numpy(dtype=float)
        row["seed_return_mean"] = float(np.mean(seed_means))
        row["seed_return_std"] = float(np.std(seed_means, ddof=0))
        row["n_seeds"] = int(len(seed_means))
    else:
        row["seed_return_mean"] = row["total_reward_mean"]
        row["seed_return_std"] = 0.0
        row["n_seeds"] = int(df["model_seed"].nunique()) if "model_seed" in df.columns else 1
    return row


def evaluate_all(args: argparse.Namespace) -> Dict[str, Any]:
    set_global_seeds(0)
    device = select_device(args.device)
    log = (lambda *a, **k: None) if args.quiet else print

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    ddqn_runs = _resolve_runs(args.models_dir, args.ddqn_runs, "ddqn", args.checkpoint)
    dqn_runs = _resolve_runs(args.models_dir, args.dqn_runs, "dqn", args.checkpoint)

    scenarios = [1, 2, 3, 4, 5]
    n_ep = int(args.episodes_per_scenario)
    envs = {s: NeonatalIncubatorEnv(scenario=s, config_path=args.config) for s in scenarios}

    # Build list of (controller_family, run_name, model_seed, policy, ckpt_val_ret)
    controllers: List[Tuple[str, str, int, Policy, float]] = [
        ("baseline", "baseline_rule_pid", -1, BaselinePolicy(name="baseline"), float("nan"))
    ]
    for run_name, ckpt_path, seed in ddqn_runs:
        ag = DDQNAgent.load(ckpt_path, device=device, seed=seed)
        val_ret = float(ag.loaded_extra.get("validation", {}).get("val_return_mean", np.nan))
        controllers.append(("ddqn", run_name, seed, AgentPolicy(ag, name=run_name), val_ret))
    for run_name, ckpt_path, seed in dqn_runs:
        ag = DDQNAgent.load(ckpt_path, device=device, seed=seed)
        val_ret = float(ag.loaded_extra.get("validation", {}).get("val_return_mean", np.nan))
        controllers.append(("dqn", run_name, seed, AgentPolicy(ag, name=run_name), val_ret))

    # Identify the best-validation DDQN and DQN checkpoints (selected on VALIDATION only)
    best_ddqn_run = None
    if ddqn_runs:
        best_ddqn_run = max(
            [c for c in controllers if c[0] == "ddqn"],
            key=lambda c: (-np.inf if np.isnan(c[4]) else c[4]),
        )[1]
    best_dqn_run = None
    if dqn_runs:
        best_dqn_run = max(
            [c for c in controllers if c[0] == "dqn"],
            key=lambda c: (-np.inf if np.isnan(c[4]) else c[4]),
        )[1]

    log("=" * 82)
    log(f"[eval] Held-out test evaluation: {len(scenarios)} scenarios x {n_ep} episodes "
        f"({len(scenarios)*n_ep} episodes/controller)")
    log(f"[eval] Controllers: baseline=1, ddqn_seeds={len(ddqn_runs)} "
        f"(selected={best_ddqn_run}), dqn_seeds={len(dqn_runs)} (selected={best_dqn_run})")
    log("=" * 82)

    ep_rows: List[Dict[str, Any]] = []
    rep_traj_rows: List[Dict[str, Any]] = []

    for family, run_name, model_seed, policy, val_ret in controllers:
        is_selected = (
            (family == "baseline")
            or (family == "ddqn" and run_name == best_ddqn_run)
            or (family == "dqn" and run_name == best_dqn_run)
        )
        for scn in scenarios:
            env = envs[scn]
            for i in range(n_ep):
                t_seed = test_seed(scn, i)
                rec_traj = (i == 0 and is_selected)
                out = run_episode(env, policy, seed=t_seed, record_trajectory=rec_traj)
                m = dict(out["metrics"])
                m.update({
                    "controller": family,
                    "run_name": run_name,
                    "model_seed": int(model_seed),
                    "is_selected_checkpoint": bool(is_selected),
                    "val_return_mean": float(val_ret),
                    "scenario": int(scn),
                    "scenario_name": SCENARIO_NAMES[scn],
                    "test_episode": int(i),
                    "test_seed": int(t_seed),
                })
                ep_rows.append(m)
                if rec_traj and "trajectory" in out:
                    for trow in out["trajectory"]:
                        r_copy = dict(trow)
                        r_copy.update({
                            "controller": family,
                            "run_name": run_name,
                            "model_seed": int(model_seed),
                            "scenario": int(scn),
                            "test_seed": int(t_seed),
                        })
                        rep_traj_rows.append(r_copy)

    for env in envs.values():
        env.close()

    ep_df = pd.DataFrame(ep_rows)
    ep_df.to_csv(os.path.join(args.output_dir, "eval_episodes.csv"), index=False)
    if rep_traj_rows:
        pd.DataFrame(rep_traj_rows).to_csv(
            os.path.join(args.output_dir, "eval_trajectories_rep.csv"), index=False
        )

    # ---- Per-scenario summary (pooled across seeds + selected-checkpoint rows)
    by_scn_rows: List[Dict[str, Any]] = []
    for (family, scn), grp in ep_df.groupby(["controller", "scenario"], sort=True):
        srow = {"controller": family, "scope": "all_seeds", "scenario": int(scn),
                "scenario_name": SCENARIO_NAMES[int(scn)]}
        srow.update(_summarise_group(grp))
        by_scn_rows.append(srow)
    for (family, scn), grp in ep_df[ep_df["is_selected_checkpoint"]].groupby(
        ["controller", "scenario"], sort=True
    ):
        if family == "baseline":
            continue
        srow = {"controller": f"{family}_selected", "scope": "selected_val_ckpt",
                "scenario": int(scn), "scenario_name": SCENARIO_NAMES[int(scn)]}
        srow.update(_summarise_group(grp))
        by_scn_rows.append(srow)
    by_scn_df = pd.DataFrame(by_scn_rows)
    by_scn_df.to_csv(os.path.join(args.output_dir, "eval_by_scenario.csv"), index=False)

    # ---- Aggregate summary across all 5 scenarios
    agg_rows: List[Dict[str, Any]] = []
    for family, grp in ep_df.groupby("controller", sort=True):
        arow = {"controller": family, "run_name": "all_seeds", "model_seed": -1}
        arow.update(_summarise_group(grp))
        agg_rows.append(arow)
    for (family, run_name, mseed), grp in ep_df.groupby(
        ["controller", "run_name", "model_seed"], sort=True
    ):
        if family == "baseline":
            continue
        arow = {"controller": f"{family}_seed", "run_name": run_name, "model_seed": int(mseed)}
        arow.update(_summarise_group(grp))
        agg_rows.append(arow)
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(os.path.join(args.output_dir, "eval_aggregate.csv"), index=False)

    # ---- Paired episode-by-episode differences on identical (scenario, test_seed)
    paired_df = _build_paired_diffs(ep_df)
    paired_df.to_csv(os.path.join(args.output_dir, "eval_paired_diff.csv"), index=False)

    # ---- Print clean summary tables
    _print_summary(by_scn_df, agg_df, paired_df, log)

    # ---- Generate all evaluation & training plots
    plot_paths = generate_all_plots(
        ep_df=ep_df,
        by_scn_df=by_scn_df,
        rep_traj_df=pd.DataFrame(rep_traj_rows) if rep_traj_rows else pd.DataFrame(),
        train_results_dir=args.train_results_dir,
        ddqn_runs=[r[0] for r in ddqn_runs],
        dqn_runs=[r[0] for r in dqn_runs],
        plots_dir=args.plots_dir,
    )

    manifest = build_manifest(
        config={
            "episodes_per_scenario": n_ep,
            "checkpoint_type": args.checkpoint,
            "ddqn_runs": [r[0] for r in ddqn_runs],
            "dqn_runs": [r[0] for r in dqn_runs],
            "best_ddqn_run_by_validation": best_ddqn_run,
            "best_dqn_run_by_validation": best_dqn_run,
        },
        seeds={
            "test_seed_rule": "900_000 + scenario*1000 + ep_idx",
            "scenarios": scenarios,
            "episodes_per_scenario": n_ep,
        },
        device=str(device),
        results={
            "aggregate": agg_df.to_dict(orient="records"),
            "by_scenario": by_scn_df.to_dict(orient="records"),
            "paired_diff": paired_df.to_dict(orient="records"),
            "plots": plot_paths,
        },
        config_path=args.config,
    )
    write_json(manifest, os.path.join(args.output_dir, "eval_manifest.json"))
    return {"ep_df": ep_df, "by_scn_df": by_scn_df, "agg_df": agg_df, "paired_df": paired_df}


def _build_paired_diffs(ep_df: pd.DataFrame) -> pd.DataFrame:
    """Compute paired episode-level differences on identical (scenario, test_seed)."""
    base = ep_df[ep_df["controller"] == "baseline"].set_index(["scenario", "test_seed"])
    rows: List[Dict[str, Any]] = []
    for target_ctrl in ("ddqn", "dqn"):
        sub = ep_df[ep_df["controller"] == target_ctrl]
        if sub.empty:
            continue
        # Average across trained seeds per (scenario, test_seed) for paired comparison
        mean_by_seed = sub.groupby(["scenario", "test_seed"])[
            ["total_reward", "tc_mean_abs_err", "spo2_mean_band_err", "control_effort", "survived"]
        ].mean()
        joined = mean_by_seed.join(
            base[["total_reward", "tc_mean_abs_err", "spo2_mean_band_err", "control_effort", "survived"]],
            lsuffix="_rl", rsuffix="_base", how="inner",
        ).reset_index()
        for scn_key, grp in list(joined.groupby("scenario")) + [("all", joined)]:
            d_ret = (grp["total_reward_rl"] - grp["total_reward_base"]).to_numpy(dtype=float)
            d_tc = (grp["tc_mean_abs_err_rl"] - grp["tc_mean_abs_err_base"]).to_numpy(dtype=float)
            d_sp = (grp["spo2_mean_band_err_rl"] - grp["spo2_mean_band_err_base"]).to_numpy(dtype=float)
            d_eff = (grp["control_effort_rl"] - grp["control_effort_base"]).to_numpy(dtype=float)
            rows.append({
                "comparison": f"{target_ctrl}_minus_baseline",
                "scenario": str(scn_key),
                "n_paired_episodes": int(len(grp)),
                "reward_diff_mean": float(np.mean(d_ret)),
                "reward_diff_std": float(np.std(d_ret, ddof=0)),
                "reward_diff_ci95": _ci95(d_ret),
                "rl_win_fraction": float(np.mean(d_ret > 0.0)),
                "tc_err_diff_mean": float(np.mean(d_tc)),
                "spo2_err_diff_mean": float(np.mean(d_sp)),
                "effort_diff_mean": float(np.mean(d_eff)),
            })
    # Also DDQN minus DQN if both present
    ddqn_sub = ep_df[ep_df["controller"] == "ddqn"]
    dqn_sub = ep_df[ep_df["controller"] == "dqn"]
    if not ddqn_sub.empty and not dqn_sub.empty:
        m_ddqn = ddqn_sub.groupby(["scenario", "test_seed"])["total_reward"].mean()
        m_dqn = dqn_sub.groupby(["scenario", "test_seed"])["total_reward"].mean()
        j = pd.DataFrame({"ddqn": m_ddqn, "dqn": m_dqn}).dropna().reset_index()
        for scn_key, grp in list(j.groupby("scenario")) + [("all", j)]:
            diff = (grp["ddqn"] - grp["dqn"]).to_numpy(dtype=float)
            rows.append({
                "comparison": "ddqn_minus_dqn",
                "scenario": str(scn_key),
                "n_paired_episodes": int(len(grp)),
                "reward_diff_mean": float(np.mean(diff)),
                "reward_diff_std": float(np.std(diff, ddof=0)),
                "reward_diff_ci95": _ci95(diff),
                "rl_win_fraction": float(np.mean(diff > 0.0)),
                "tc_err_diff_mean": np.nan,
                "spo2_err_diff_mean": np.nan,
                "effort_diff_mean": np.nan,
            })
    return pd.DataFrame(rows)


def _print_summary(by_scn_df: pd.DataFrame, agg_df: pd.DataFrame,
                   paired_df: pd.DataFrame, log) -> None:
    log("\n--- Per-Scenario Held-Out Evaluation (mean +/- std) ---")
    cols = [
        "controller", "scenario", "total_reward_mean", "total_reward_std",
        "tc_mean_abs_err_mean", "spo2_mean_band_err_mean",
        "control_effort_mean", "survived_mean", "episode_length_mean",
    ]
    sub = by_scn_df[by_scn_df["scope"] == "all_seeds"][cols]
    with pd.option_context("display.max_columns", None, "display.width", 140,
                           "display.float_format", lambda v: f"{v:8.3f}"):
        log(sub.to_string(index=False))

    log("\n--- Aggregate Across All 5 Scenarios ---")
    acols = [
        "controller", "run_name", "n_seeds", "total_reward_mean", "total_reward_std",
        "seed_return_std", "tc_mean_abs_err_mean", "spo2_mean_band_err_mean",
        "control_effort_mean", "survived_mean",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 140,
                           "display.float_format", lambda v: f"{v:8.3f}"):
        log(agg_df[acols].to_string(index=False))

    if not paired_df.empty:
        log("\n--- Paired Differences on Identical (Scenario, Seed) Episodes ---")
        with pd.option_context("display.max_columns", None, "display.width", 140,
                               "display.float_format", lambda v: f"{v:8.3f}"):
            log(paired_df.to_string(index=False))


# --------------------------------------------------------------------------- #
#  Plotting                                                                   #
# --------------------------------------------------------------------------- #
def generate_all_plots(
    ep_df: pd.DataFrame,
    by_scn_df: pd.DataFrame,
    rep_traj_df: pd.DataFrame,
    train_results_dir: str,
    ddqn_runs: List[str],
    dqn_runs: List[str],
    plots_dir: str,
) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: List[str] = []
    colors = {"baseline": "#2b5c8f", "ddqn": "#d95f02", "dqn": "#7570b3",
              "ddqn_selected": "#e7298a"}

    # 1. Learning curves & validation curves across seeds
    p1 = os.path.join(plots_dir, "ddqn_learning_curves.png")
    if _plot_learning_curves(train_results_dir, ddqn_runs, dqn_runs, p1, plt, colors):
        written.append(p1)

    # 2. Loss, Q-value, TD-error, and grad-norm diagnostics
    p2 = os.path.join(plots_dir, "ddqn_loss_and_q_diagnostics.png")
    if _plot_loss_and_q(train_results_dir, ddqn_runs, dqn_runs, p2, plt, colors):
        written.append(p2)

    # 3. Per-scenario comparison bar chart (Baseline vs DDQN vs DQN)
    p3 = os.path.join(plots_dir, "ddqn_vs_baseline_by_scenario.png")
    _plot_by_scenario_bars(by_scn_df, p3, plt, colors)
    written.append(p3)

    # 4. Seed variability plot
    p4 = os.path.join(plots_dir, "ddqn_seed_variability.png")
    _plot_seed_variability(ep_df, p4, plt, colors)
    written.append(p4)

    # 5. Representative time-series trajectories across all 5 scenarios
    if not rep_traj_df.empty:
        p5 = os.path.join(plots_dir, "ddqn_timeseries_scenarios.png")
        _plot_timeseries(rep_traj_df, p5, plt, colors)
        written.append(p5)

        p6 = os.path.join(plots_dir, "ddqn_action_timelines.png")
        _plot_action_timelines(rep_traj_df, ep_df, p6, plt, colors)
        written.append(p6)

    return written


def _load_run_csvs(train_results_dir: str, runs: List[str], fname: str) -> List[pd.DataFrame]:
    dfs = []
    for r in runs:
        path = os.path.join(train_results_dir, r, fname)
        if os.path.isfile(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    df["run_name"] = r
                    dfs.append(df)
            except Exception:
                pass
    return dfs


def _interp_curve(dfs: List[pd.DataFrame], xcol: str, ycol: str,
                  n_grid: int = 100, rolling: int = 1) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not dfs:
        return None
    xmax = min(float(df[xcol].max()) for df in dfs if xcol in df and ycol in df)
    xmin = max(float(df[xcol].min()) for df in dfs if xcol in df and ycol in df)
    if xmax <= xmin:
        return None
    grid = np.linspace(xmin, xmax, n_grid)
    ys = []
    for df in dfs:
        s = df[ycol].rolling(rolling, min_periods=1).mean().to_numpy(dtype=float)
        x = df[xcol].to_numpy(dtype=float)
        ys.append(np.interp(grid, x, s))
    arr = np.vstack(ys)
    return grid, arr.mean(axis=0), arr.std(axis=0)


def _plot_learning_curves(train_results_dir: str, ddqn_runs: List[str],
                          dqn_runs: List[str], out_path: str, plt, colors) -> bool:
    ddqn_ep = _load_run_csvs(train_results_dir, ddqn_runs, "episodes.csv")
    dqn_ep = _load_run_csvs(train_results_dir, dqn_runs, "episodes.csv")
    ddqn_val = _load_run_csvs(train_results_dir, ddqn_runs, "validation.csv")
    dqn_val = _load_run_csvs(train_results_dir, dqn_runs, "validation.csv")
    if not ddqn_ep and not dqn_ep:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel A: Smoothed training episode return vs env_step
    ax = axes[0]
    for label, dfs, col in [("DDQN", ddqn_ep, colors["ddqn"]), ("DQN", dqn_ep, colors["dqn"])]:
        res = _interp_curve(dfs, "env_step", "return", n_grid=120, rolling=25)
        if res is not None:
            g, m, s = res
            ax.plot(g / 1000.0, m, label=f"{label} (n={len(dfs)} seeds)", color=col, lw=2)
            ax.fill_between(g / 1000.0, m - s, m + s, color=col, alpha=0.2)
    ax.set_xlabel("Environment steps (x1000)")
    ax.set_ylabel("Training episode return (25-ep rolling)")
    ax.set_title("A. Training Return Across Seeds")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=9)

    # Panel B: Periodic greedy validation return vs env_step
    ax = axes[1]
    for label, dfs, col in [("DDQN", ddqn_val, colors["ddqn"]), ("DQN", dqn_val, colors["dqn"])]:
        res = _interp_curve(dfs, "env_step", "val_return_mean", n_grid=60, rolling=1)
        if res is not None:
            g, m, s = res
            ax.plot(g / 1000.0, m, label=f"{label} greedy val", color=col, lw=2, marker="o", ms=3)
            ax.fill_between(g / 1000.0, m - s, m + s, color=col, alpha=0.2)
    ax.axhline(-23.04, color=colors["baseline"], ls="--", lw=1.6, label="Baseline mean (-23.04)")
    ax.set_xlabel("Environment steps (x1000)")
    ax.set_ylabel("Validation return (5-scenario mean)")
    ax.set_title("B. Greedy Validation Return (Mean +/- 1 SD)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=9)

    # Panel C: Safety early-termination rate during training
    ax = axes[2]
    for label, dfs, col in [("DDQN", ddqn_ep, colors["ddqn"]), ("DQN", dqn_ep, colors["dqn"])]:
        for df in dfs:
            df["term_float"] = df["terminated"].astype(float)
        res = _interp_curve(dfs, "env_step", "term_float", n_grid=120, rolling=50)
        if res is not None:
            g, m, s = res
            ax.plot(g / 1000.0, m * 100.0, label=f"{label}", color=col, lw=2)
            ax.fill_between(g / 1000.0, np.clip((m - s) * 100.0, 0, 100),
                            np.clip((m + s) * 100.0, 0, 100), color=col, alpha=0.2)
    ax.set_xlabel("Environment steps (x1000)")
    ax.set_ylabel("Early safety termination rate (%)")
    ax.set_title("C. Exploration Safety-Termination Rate (50-ep rolling)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("DDQN vs Standard DQN Multi-Seed Learning Progression (Simulation-Only)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _plot_loss_and_q(train_results_dir: str, ddqn_runs: List[str],
                     dqn_runs: List[str], out_path: str, plt, colors) -> bool:
    ddqn_up = _load_run_csvs(train_results_dir, ddqn_runs, "updates.csv")
    dqn_up = _load_run_csvs(train_results_dir, dqn_runs, "updates.csv")
    if not ddqn_up and not dqn_up:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    specs = [
        (axes[0, 0], "loss", "Huber TD Loss", "A. Training Huber Loss"),
        (axes[0, 1], "q_mean", "Mean Predicted Q(s, a) (scaled)", "B. Mean Predicted Action-Value Q (Overestimation Check)"),
        (axes[1, 0], "td_abs_mean", "Mean |TD Error|", "C. Mean Absolute TD Error"),
        (axes[1, 1], "grad_norm", "Clipped Gradient Norm", "D. Parameter Gradient Norm"),
    ]
    for ax, col_name, ylabel, title in specs:
        for label, dfs, col in [("DDQN", ddqn_up, colors["ddqn"]), ("DQN", dqn_up, colors["dqn"])]:
            res = _interp_curve(dfs, "env_step", col_name, n_grid=100, rolling=3)
            if res is not None:
                g, m, s = res
                ax.plot(g / 1000.0, m, label=f"{label}", color=col, lw=2)
                ax.fill_between(g / 1000.0, m - s, m + s, color=col, alpha=0.2)
        ax.set_xlabel("Environment steps (x1000)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("DDQN vs Standard DQN Optimization & Q-Value Stability Diagnostics", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _plot_by_scenario_bars(by_scn_df: pd.DataFrame, out_path: str, plt, colors) -> None:
    sub = by_scn_df[by_scn_df["scope"] == "all_seeds"].copy()
    controllers = [c for c in ("baseline", "ddqn", "dqn") if c in sub["controller"].unique()]
    scenarios = sorted(sub["scenario"].unique())
    x = np.arange(len(scenarios))
    width = 0.8 / max(1, len(controllers))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [
        (axes[0, 0], "total_reward", "Cumulative Episode Reward", "A. Cumulative Reward by Scenario (Higher = Better)"),
        (axes[0, 1], "tc_mean_abs_err", "Mean |Tc - 37.0| (deg C)", "B. Core Temperature Tracking Error (Lower = Better)"),
        (axes[1, 0], "spo2_mean_band_err", "Mean SpO2 Band Error (%)", "C. SpO2 Out-of-Band [91, 95]% Error (Lower = Better)"),
        (axes[1, 1], "control_effort", "Mean Control Effort (u_heater + u_fio2)", "D. Actuator Control Effort (Lower = More Conservative)"),
    ]
    labels_map = {"baseline": "Rule-Based Baseline", "ddqn": "DDQN (Multi-Seed)", "dqn": "Standard DQN (Multi-Seed)"}

    for ax, metric, ylabel, title in panels:
        for idx, ctrl in enumerate(controllers):
            c_df = sub[sub["controller"] == ctrl].set_index("scenario").reindex(scenarios)
            means = c_df[f"{metric}_mean"].to_numpy(dtype=float)
            stds = c_df[f"{metric}_std"].to_numpy(dtype=float)
            offset = (idx - (len(controllers) - 1) / 2.0) * width
            ax.bar(
                x + offset, means, width=width * 0.92, yerr=stds, capsize=3,
                label=labels_map.get(ctrl, ctrl), color=colors.get(ctrl, "#555555"),
                edgecolor="black", linewidth=0.6, alpha=0.9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{s}: {SCENARIO_NAMES[s][:18]}" for s in scenarios], rotation=12, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=8.5)

    fig.suptitle("Held-Out Test Evaluation Across Scenarios (Identical Seeds, Mean +/- 1 SD)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_seed_variability(ep_df: pd.DataFrame, out_path: str, plt, colors) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Aggregate return per trained seed
    ax = axes[0]
    base_ret = float(ep_df[ep_df["controller"] == "baseline"]["total_reward"].mean())
    ax.axhline(base_ret, color=colors["baseline"], ls="--", lw=2, label=f"Baseline ({base_ret:.2f})")

    x_ticks = []
    x_labels = []
    pos = 0
    for ctrl in ("ddqn", "dqn"):
        sub = ep_df[ep_df["controller"] == ctrl]
        for seed, grp in sub.groupby("model_seed", sort=True):
            m = float(grp["total_reward"].mean())
            s = float(grp["total_reward"].std(ddof=0))
            ax.bar(pos, m, yerr=s, capsize=4, color=colors.get(ctrl, "#666"),
                   edgecolor="black", alpha=0.88,
                   label=ctrl.upper() if seed == sub["model_seed"].min() else None)
            x_ticks.append(pos)
            x_labels.append(f"{ctrl.upper()}\nseed {seed}")
            pos += 1
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Held-Out Return Across All 5 Scenarios")
    ax.set_title("A. Overall Return by Trained Seed (Mean +/- 1 SD)")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)

    # Right: Scenario 5 (desaturation challenge) return per trained seed
    ax = axes[1]
    s5_base = float(ep_df[(ep_df["controller"] == "baseline") & (ep_df["scenario"] == 5)]["total_reward"].mean())
    ax.axhline(s5_base, color=colors["baseline"], ls="--", lw=2, label=f"Baseline S5 ({s5_base:.2f})")
    pos = 0
    x_ticks_5 = []
    for ctrl in ("ddqn", "dqn"):
        sub = ep_df[(ep_df["controller"] == ctrl) & (ep_df["scenario"] == 5)]
        for seed, grp in sub.groupby("model_seed", sort=True):
            m = float(grp["total_reward"].mean())
            s = float(grp["total_reward"].std(ddof=0))
            ax.bar(pos, m, yerr=s, capsize=4, color=colors.get(ctrl, "#666"),
                   edgecolor="black", alpha=0.88,
                   label=ctrl.upper() if seed == sub["model_seed"].min() else None)
            x_ticks_5.append(pos)
            pos += 1
    ax.set_xticks(x_ticks_5)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Scenario 5 Held-Out Return")
    ax.set_title("B. Scenario 5 (Desaturation) Return by Trained Seed")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)

    fig.suptitle("Multi-Seed Reproducibility on Held-Out Evaluation Episodes", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_timeseries(rep_traj_df: pd.DataFrame, out_path: str, plt, colors) -> None:
    scenarios = sorted(rep_traj_df["scenario"].unique())
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(14, 2.7 * len(scenarios)), sharex=True)
    if len(scenarios) == 1:
        axes = np.array([axes])

    for row_idx, scn in enumerate(scenarios):
        ax_tc, ax_sp = axes[row_idx, 0], axes[row_idx, 1]
        sub_s = rep_traj_df[rep_traj_df["scenario"] == scn]
        for ctrl in ("baseline", "ddqn", "dqn"):
            c_df = sub_s[sub_s["controller"] == ctrl]
            if c_df.empty:
                continue
            lw = 2.0 if ctrl != "dqn" else 1.4
            ls = "-" if ctrl != "dqn" else ":"
            ax_tc.plot(c_df["step"], c_df["Tc"], label=ctrl.upper(),
                       color=colors.get(ctrl, "#333"), lw=lw, ls=ls)
            ax_sp.plot(c_df["step"], c_df["SpO2"], label=ctrl.upper(),
                       color=colors.get(ctrl, "#333"), lw=lw, ls=ls)

        ax_tc.axhline(37.0, color="green", ls="--", lw=0.9, alpha=0.7)
        ax_tc.axhspan(36.5, 37.5, color="green", alpha=0.08)
        ax_tc.axhline(35.5, color="red", ls=":", lw=0.9, alpha=0.7)
        ax_tc.axhline(38.0, color="red", ls=":", lw=0.9, alpha=0.7)
        ax_tc.set_ylabel(f"S{scn} Tc (deg C)", fontsize=9)
        ax_tc.set_title(f"Scenario {scn} ({SCENARIO_NAMES[scn]}) -- Core Temperature Tc", fontsize=9.5)
        ax_tc.grid(True, alpha=0.3, linestyle="--")

        ax_sp.axhspan(91.0, 95.0, color="green", alpha=0.08)
        ax_sp.axhline(91.0, color="green", ls="--", lw=0.9, alpha=0.7)
        ax_sp.axhline(95.0, color="green", ls="--", lw=0.9, alpha=0.7)
        ax_sp.axhline(85.0, color="red", ls=":", lw=0.9, alpha=0.7)
        ax_sp.axhline(96.0, color="red", ls=":", lw=0.9, alpha=0.7)
        ax_sp.set_ylabel(f"S{scn} SpO2 (%)", fontsize=9)
        ax_sp.set_title(f"Scenario {scn} ({SCENARIO_NAMES[scn]}) -- Arterial SpO2", fontsize=9.5)
        ax_sp.grid(True, alpha=0.3, linestyle="--")
        if row_idx == 0:
            ax_tc.legend(loc="best", fontsize=8)
            ax_sp.legend(loc="best", fontsize=8)

    axes[-1, 0].set_xlabel("Control Step (45 s / step)")
    axes[-1, 1].set_xlabel("Control Step (45 s / step)")
    fig.suptitle("Held-Out Test Episode 0 Trajectories: Baseline vs DDQN vs DQN", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_action_timelines(rep_traj_df: pd.DataFrame, ep_df: pd.DataFrame,
                           out_path: str, plt, colors) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))

    # Top row: Scenario 3 (porthole) heater fraction & Scenario 5 (desat) FiO2 command
    ax_s3 = axes[0, 0]
    s3 = rep_traj_df[(rep_traj_df["scenario"] == 3) & (rep_traj_df["step"] >= 1)]
    for ctrl in ("baseline", "ddqn", "dqn"):
        c_df = s3[s3["controller"] == ctrl]
        if not c_df.empty:
            ax_s3.step(c_df["step"], c_df["u_heater"], where="post",
                       label=ctrl.upper(), color=colors.get(ctrl, "#333"), lw=1.6, alpha=0.85)
    ax_s3.axvspan(50, 80, color="orange", alpha=0.15, label="Porthole open [50,80]")
    ax_s3.set_xlabel("Control Step")
    ax_s3.set_ylabel("Heater Fraction u_heater")
    ax_s3.set_title("A. Scenario 3 (Porthole) Heater Power Timeline")
    ax_s3.grid(True, alpha=0.3, linestyle="--")
    ax_s3.legend(loc="best", fontsize=8.5)

    ax_s5 = axes[0, 1]
    s5 = rep_traj_df[(rep_traj_df["scenario"] == 5) & (rep_traj_df["step"] >= 1)]
    for ctrl in ("baseline", "ddqn", "dqn"):
        c_df = s5[s5["controller"] == ctrl]
        if not c_df.empty:
            ax_s5.step(c_df["step"], c_df["fio2_cmd"] / 100.0 if c_df["fio2_cmd"].max() > 1.5 else c_df["fio2_cmd"],
                       where="post", label=ctrl.upper(), color=colors.get(ctrl, "#333"), lw=1.6, alpha=0.85)
    ax_s5.axvspan(60, 120, color="purple", alpha=0.12, label="Desaturation [60,120]")
    ax_s5.set_xlabel("Control Step")
    ax_s5.set_ylabel("Commanded FiO2 (volume fraction)")
    ax_s5.set_title("B. Scenario 5 (Desaturation) FiO2 Command Timeline")
    ax_s5.grid(True, alpha=0.3, linestyle="--")
    ax_s5.legend(loc="best", fontsize=8.5)

    # Bottom left: Action switch count per episode by scenario
    ax_sw = axes[1, 0]
    scenarios = sorted(ep_df["scenario"].unique())
    x = np.arange(len(scenarios))
    ctrls = [c for c in ("baseline", "ddqn", "dqn") if c in ep_df["controller"].unique()]
    w = 0.8 / max(1, len(ctrls))
    for i, ctrl in enumerate(ctrls):
        means = [ep_df[(ep_df["controller"] == ctrl) & (ep_df["scenario"] == s)]["action_switches"].mean()
                 for s in scenarios]
        offset = (i - (len(ctrls) - 1) / 2.0) * w
        ax_sw.bar(x + offset, means, width=w * 0.9, label=ctrl.upper(),
                  color=colors.get(ctrl, "#333"), edgecolor="black", alpha=0.9)
    ax_sw.set_xticks(x)
    ax_sw.set_xticklabels([f"S{s}" for s in scenarios])
    ax_sw.set_xlabel("Scenario")
    ax_sw.set_ylabel("Mean Action Switches / 200-step Episode")
    ax_sw.set_title("C. Actuator Switching Frequency (Chattering Check)")
    ax_sw.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax_sw.legend(loc="best", fontsize=8.5)

    # Bottom right: Heater energy (kJ) per episode by scenario
    ax_en = axes[1, 1]
    for i, ctrl in enumerate(ctrls):
        means = [ep_df[(ep_df["controller"] == ctrl) & (ep_df["scenario"] == s)]["heater_energy_kJ"].mean()
                 for s in scenarios]
        offset = (i - (len(ctrls) - 1) / 2.0) * w
        ax_en.bar(x + offset, means, width=w * 0.9, label=ctrl.upper(),
                  color=colors.get(ctrl, "#333"), edgecolor="black", alpha=0.9)
    ax_en.set_xticks(x)
    ax_en.set_xticklabels([f"S{s}" for s in scenarios])
    ax_en.set_xlabel("Scenario")
    ax_en.set_ylabel("Mean Heater Energy per Episode (kJ)")
    ax_en.set_title("D. Cumulative Heater Energy Consumption (kJ)")
    ax_en.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax_en.legend(loc="best", fontsize=8.5)

    fig.suptitle("Actuator Timelines, Switching Frequency & Energy Usage", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    evaluate_all(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())

