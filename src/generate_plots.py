#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 Neonatal Incubator -- Baseline Performance Plot Generator (Member 1)
------------------------------------------------------------------------------
Project : RL + XAI for Neonatal Incubator Temperature & Oxygen Regulation
          using DDQN + SHAP.
Member  : 1 (Simulation Environment, Scenarios & Baseline Visualisation)
Module  : generate_plots.py
Purpose : Read the per-scenario trajectory CSVs produced by the updated
          ``run_simulation.py`` (located in ``results/``) and render the
          baseline-performance plots into ``plots/``.

This is Member 1's baseline-visualisation deliverable.  Member 2 will later
overlay the DDQN learning curves (episode reward, length, loss, evaluation
return per scenario) into the same plot directory for side-by-side
baseline-vs-learned-policy comparison, and Member 3 will add SHAP-based
feature-attribution overlays.

Input CSV layout (NEW 11-dim observation schema)
-----------------------------------------------
``run_simulation.py`` writes, per scenario ``S`` and episode ``E``::

    trajectory_scenario{S}_ep{E}.csv   # one CSV per (scenario, episode)
    summary.csv                        # one row per (scenario, episode)

Each trajectory CSV has columns (the NEW schema produced by the updated
``run_simulation.py``)::

    step, action, heater_idx, o2_idx, u_heater, fio2_cmd, Q_heater_W,
    Tc, Tsk, Ta, C_O2, SpO2, Tamb, dTa_dt, dC_O2_dt,
    phase, disturbance,
    reward, r_thermal, r_oxygen, r_energy, r_safety,
    terminated, truncated

Key differences from the legacy schema:

* ``u_o2`` (normalized [0,1]) is replaced by ``fio2_cmd`` -- the absolute
  inspired-oxygen fraction in ``[0.21, 0.60]`` (room air to high-flow O2).
* ``Q_heater_W`` is the electrical heater power in watts
  (``u_heater * Q_HEATER_MAX`` with ``Q_HEATER_MAX = 360 W``).
* ``phase`` is the normalised time-in-episode ``step / max_steps`` in
  ``[0, 1]`` (one of the two new observation dimensions).
* ``disturbance`` is the integer disturbance code exposed in the
  observation (the second new dimension):

      0  ->  idle                (no disturbance active)
      1  ->  porthole open       (Scenario 3, convective heat-loss spike)
      2  ->  O2 leak active      (Scenario 5, inspired-O2 mixing perturbed)

  When ``disturbance > 0`` the corresponding steps are shaded as light-red
  vertical bands on every per-step plot so the disturbance window is
  visually obvious against the controller response.
* ``reward_thermal`` / ``reward_oxygen`` / ``reward_energy`` /
  ``reward_safety`` are renamed to ``r_thermal`` / ``r_oxygen`` /
  ``r_energy`` / ``r_safety`` (matching the env's info-dict keys).
* ``done`` is split into the Gymnasium pair ``terminated`` (hard safety
  breach) and ``truncated`` (time limit reached).  These flags drive the
  per-scenario episode-length statistics.

For backwards compatibility the loader ALSO accepts the legacy single
per-scenario file ``trajectory_scenario{S}.csv`` (one episode per file).

Aggregation
-----------
Because the evaluation protocol (``docs/scenarios.md`` §7) runs several
episodes per scenario and asks for "mean +/- 1 std", every per-scenario
trajectory plot renders the **mean** across episodes as a solid line
with a translucent **+/-1 sigma** band.  When only a single episode is
available the band collapses to zero width and the plot reduces to a
plain trajectory.  The cross-scenario summary bar chart reports the
per-scenario mean of each headline metric, with +/-1 sigma error bars
when >= 2 episodes exist; values are read from ``summary.csv`` when
present and otherwise recomputed from the raw trajectories.

Plots generated
---------------
Per scenario ``S`` (one PNG each):

    1. ``temperature_trajectory_scenario{S}.png``
       Mean Tc, Tsk, Ta, Tamb (+/-1 sigma band) over steps with horizontal
       reference lines for the core-temperature target (37 deg C) and the
       safety bounds (35.5 / 38.0 deg C).  Disturbance windows shaded.
    2. ``oxygen_trajectory_scenario{S}.png``
       Mean SpO2 (top) and C_O2 (bottom) over steps with reference lines
       for the reward target band (91 / 95 %) and the hard safety bounds
       (85 / 96 %).  Disturbance windows shaded.
    3. ``reward_trajectory_scenario{S}.png``
       Per-step reward (top) and cumulative reward (bottom), mean +/- 1
       sigma.  Disturbance windows shaded.
    4. ``control_actions_scenario{S}.png``
       Heater level ``u_heater`` (top) and absolute FiO2 command
       ``fio2_cmd`` (bottom) over steps, mean +/- 1 sigma.

Cross-scenario summaries (one PNG each):

    5. ``summary_bar.png``
       2x2 bar chart comparing the scenarios on: total_reward,
       mean |Tc - 37|, mean |SpO2 - 92|, and number of safety violations
       (per-episode mean, +/-1 sigma error bars when >= 2 episodes).
    6. ``scenario_comparison_temperature.png``
       Mean Tc trajectory per scenario overlaid on one axis.
    7. ``scenario_comparison_spo2.png``
       Mean SpO2 trajectory per scenario overlaid on one axis.

Robustness
----------
* Missing results directory / missing scenario CSV -> prints a warning
  and skips that scenario (the script still exits 0 and produces
  whatever plots it can).
* Missing ``cumulative_reward`` -> derived as ``reward.cumsum()``.
* Missing ``safe`` flag -> re-derived from the Tc / SpO2 safety bounds.
* Missing ``fio2_cmd`` -> derived from ``o2_idx`` via the env's frozen
  FIO2_LEVELS grid (0.21, 0.30, 0.40, 0.50, 0.60) so the FiO2 subplot
  still renders on legacy CSVs that only logged ``u_o2``.
* Missing ``disturbance`` -> no shading (graceful no-op).
* Varying episode counts per scenario -> NaN-padded aggregation so the
  common prefix is well-defined and the tail degrades gracefully.

Usage
-----
::

    python generate_plots.py
    python generate_plots.py --results-dir results --plots-dir plots \\
        --scenarios 1,2,3,4,5

==============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # noqa: E402  must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams['axes.unicode_minus'] = False

# --------------------------------------------------------------------------- #
#  Domain constants -- mirror NeonatalIncubatorEnv (environment.py) and the
#  safety bounds in config.yaml.  Single source of truth for all reference
#  lines drawn on the plots.
# --------------------------------------------------------------------------- #
TARGET_TC: float = 37.0           # core-temperature setpoint (deg C)
TC_SAFETY_LOWER: float = 35.5     # hard hypothermia -> terminated
TC_SAFETY_UPPER: float = 38.0     # hard hyperthermia -> terminated

# SpO2 reward uses a target BAND [91, 95]; safety band is [85, 96].
SPO2_TARGET_LOWER: float = 91.0
SPO2_TARGET_UPPER: float = 95.0
SPO2_SAFETY_LOWER: float = 85.0
SPO2_SAFETY_UPPER: float = 96.0
# Midpoint of the target band -- used for the |SpO2 - target| metric only.
TARGET_SPO2: float = 92.0

# Discrete-action decoding (matches environment.py / mdp_specification).
# heater levels are fractional [0,1]; FiO2 levels are ABSOLUTE fractions
# in [0.21, 0.60] (not normalised [0,1]).
HEATER_LEVELS: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
FIO2_LEVELS: Tuple[float, ...] = (0.21, 0.30, 0.40, 0.50, 0.60)
N_O2_LEVELS: int = len(FIO2_LEVELS)

# Human-readable scenario labels (canonical names come from
# environment.py SCENARIO_NAMES; rendered in title case for plot titles).
SCENARIO_NAMES: Dict[int, str] = {
    1: "Normal standard",
    2: "Ambient temperature drop",
    3: "Porthole opening event",
    4: "Initial hypothermia",
    5: "Oxygen desaturation event",
}

# Disturbance code -> human-readable label (matches env's disturbance_flag).
DISTURBANCE_LABELS: Dict[int, str] = {
    1: "Porthole open",
    2: "O2 leak",
}

# Consistent colour cycle (tab10) used for the scenario overlays and bars.
COLOR_CYCLE = list(plt.get_cmap("tab10").colors)

# Per-signal colours for the multi-line trajectory plots.
TEMP_COLORS: Dict[str, str] = {
    "Tc": "#d62728",    # red    -- core temperature (primary)
    "Tsk": "#ff7f0e",   # orange -- skin temperature
    "Ta": "#2ca02c",    # green  -- incubator air temperature
    "Tamb": "#17becf",  # teal   -- ambient room temperature
}
O2_COLORS: Dict[str, str] = {
    "SpO2": "#1f77b4",  # blue   -- saturation
    "C_O2": "#9467bd",  # purple -- inspired O2 concentration
}

# Disturbance shading colour (light red, used for axvspan bands).
DISTURBANCE_COLOR: str = "red"
DISTURBANCE_ALPHA: float = 0.10


# --------------------------------------------------------------------------- #
#  CSV reading & per-episode normalisation
# --------------------------------------------------------------------------- #
def _column(df: pd.DataFrame, *candidates) -> Optional[pd.Series]:
    """Return the first existing column from ``candidates`` else ``None``."""
    for name in candidates:
        if name in df.columns:
            return df[name]
    return None


def _decode_action(action) -> Tuple[float, float]:
    """Decode a discrete action ``a in {0..24}`` into ``(u_heater, fio2_cmd)``.

    Uses ``heater_idx = a // 5`` and ``o2_idx = a % 5`` then maps to the
    fractional command levels.  ``u_heater`` is the heater duty cycle in
    [0, 1]; ``fio2_cmd`` is the absolute inspired-oxygen fraction in
    [0.21, 0.60].  Robust to NaN / non-int / out-of-range inputs (returns
    NaN for the offending component so matplotlib skips it).
    """
    try:
        a = int(round(float(action)))
    except (TypeError, ValueError):
        return float("nan"), float("nan")
    if a < 0 or a >= len(HEATER_LEVELS) * N_O2_LEVELS:
        return float("nan"), float("nan")
    heater_idx = a // N_O2_LEVELS
    o2_idx = a % N_O2_LEVELS
    return float(HEATER_LEVELS[heater_idx]), float(FIO2_LEVELS[o2_idx])


def _normalize_episode(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical derived columns exist on a single-episode DataFrame.

    Guarantees: ``step``, ``cumulative_reward``, ``safe``, ``fio2_cmd``
    (when derivable).  Derived from sibling columns when absent.  Returns
    the same DataFrame (mutated in place for efficiency).
    """
    if "step" not in df.columns:
        df["step"] = np.arange(len(df))

    # Decoded FiO2 command: prefer the explicit ``fio2_cmd`` column (NEW
    # schema, absolute fraction 0.21-0.60); otherwise decode from the
    # discrete ``action`` column (which gracefully yields NaN for the
    # initial-state row where action == -1); final fallback is the legacy
    # ``u_o2`` column (normalised [0,1]) mapped back through the env's
    # FIO2_LEVELS grid.  We always end up writing the value into
    # ``fio2_cmd`` so the rest of the pipeline has one canonical column
    # name.
    if "fio2_cmd" not in df.columns:
        if _column(df, "fio2_cmd") is not None:
            pass  # present under a slightly different case-sensitive name
        elif "action" in df.columns:
            df["fio2_cmd"] = df["action"].apply(
                lambda a: _decode_action(a)[1])
        else:
            u_o2 = _column(df, "u_o2")
            if u_o2 is not None:
                # Legacy: u_o2 is the normalised [0,1] index into the
                # 5-level FIO2 grid.  Map back to absolute fraction,
                # preserving NaN (initial-state row).
                vals = u_o2.astype(float).to_numpy()
                nan_mask = ~np.isfinite(vals)
                idx = np.clip(
                    np.rint(np.nan_to_num(vals, nan=0.0)
                            * (N_O2_LEVELS - 1)),
                    0, N_O2_LEVELS - 1,
                ).astype(np.intp)
                fio2 = np.asarray(FIO2_LEVELS, dtype=float)[idx]
                fio2 = fio2.astype(float)
                fio2[nan_mask] = np.nan
                df["fio2_cmd"] = fio2
            else:
                df["fio2_cmd"] = float("nan")

    # Cumulative reward for the bottom panel of the reward-plot.
    if "cumulative_reward" not in df.columns and "reward" in df.columns:
        df["cumulative_reward"] = df["reward"].cumsum()

    # ``safe`` flag for safety-violation counting.
    if "safe" not in df.columns and "Tc" in df.columns \
            and "SpO2" in df.columns:
        tc = df["Tc"].astype(float).values
        spo2 = df["SpO2"].astype(float).values
        unsafe = (
            (tc < TC_SAFETY_LOWER) | (tc > TC_SAFETY_UPPER)
            | (spo2 < SPO2_SAFETY_LOWER) | (spo2 > SPO2_SAFETY_UPPER)
        )
        df["safe"] = ~unsafe

    # Disturbance column is OPTIONAL -- if missing, treat as all-idle so
    # the shading helper is a graceful no-op.
    if "disturbance" not in df.columns:
        df["disturbance"] = 0

    return df


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read & normalise one trajectory CSV.  ``None`` on failure (warn)."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover -- defensive
        print(f"[WARN] Failed to read {path}: {exc} -- skipping.",
              file=sys.stderr)
        return None
    if df.empty:
        print(f"[WARN] Trajectory CSV is empty: {path} -- skipping.",
              file=sys.stderr)
        return None
    return _normalize_episode(df)


def load_scenario_episodes(results_dir: Path, scenario: int) -> List[pd.DataFrame]:
    """Load every episode DataFrame for one scenario.

    Resolution order (first match wins):
      1. ``trajectory_scenario{S}.csv``  -- single per-scenario file
         (legacy / minimal schema); loaded as a one-element list.
      2. ``trajectory_scenario{S}_ep*.csv`` -- per-episode files written by
         ``run_simulation.py``; sorted by globbed filename for deterministic
         aggregation.

    Returns an empty list (after a warning) if no CSV matches.
    """
    single = results_dir / f"trajectory_scenario{scenario}.csv"
    if os.path.exists(single):
        df = _read_csv(single)
        return [df] if df is not None else []

    ep_paths = sorted(
        results_dir.glob(f"trajectory_scenario{scenario}_ep*.csv"))
    dfs: List[pd.DataFrame] = []
    for p in ep_paths:
        df = _read_csv(p)
        if df is not None:
            dfs.append(df)
    if not dfs:
        print(f"[WARN] No trajectory CSVs for scenario {scenario} under "
              f"{results_dir} (looked for trajectory_scenario{scenario}.csv"
              f" and trajectory_scenario{scenario}_ep*.csv) -- skipping.",
              file=sys.stderr)
    return dfs


# --------------------------------------------------------------------------- #
#  Per-episode aggregation helpers
# --------------------------------------------------------------------------- #
def _aligned_stats(dfs: Sequence[pd.DataFrame],
                   col: str) -> Tuple[Optional[np.ndarray],
                                      Optional[np.ndarray],
                                      Optional[np.ndarray]]:
    """Return ``(steps, mean, std)`` for ``col`` across episodes.

    Episodes of unequal length are right-padded with NaN; ``mean`` / ``std``
    are computed with NaN-aware reductions so the common prefix (where all
    episodes have data) is well-defined and the tail degrades gracefully.
    ``std`` is zero everywhere when only one episode is present.  Returns
    ``(None, None, None)`` when the column is absent from every episode.
    """
    series = [df[col].reset_index(drop=True) for df in dfs if col in df.columns]
    if not series:
        return None, None, None
    maxlen = max(len(s) for s in series)
    mat = np.full((maxlen, len(series)), np.nan, dtype=float)
    for j, s in enumerate(series):
        vals = s.to_numpy(dtype=float)
        mat[:len(vals), j] = vals
    steps = np.arange(maxlen)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore", all="ignore"):
            mean = np.nanmean(mat, axis=1)
            if mat.shape[1] > 1:
                std = np.nanstd(mat, axis=1, ddof=0)
            else:
                std = np.zeros(maxlen, dtype=float)
    return steps, mean, std


def _band(ax, steps, mean, std, color, label=None) -> None:
    """Draw a mean line with a translucent +/-1 sigma band.  No-op if
    ``mean`` is ``None``."""
    if mean is None:
        return
    ax.plot(steps, mean, color=color, linewidth=1.6, label=label)
    if std is not None and np.any(np.isfinite(std)) and np.nanmax(std) > 0:
        ax.fill_between(steps, mean - std, mean + std, color=color,
                        alpha=0.15, linewidth=0)


def _disturbance_mask(dfs: Sequence[pd.DataFrame]) -> Optional[np.ndarray]:
    """Return a 1-D boolean mask of "any disturbance active" per step.

    Aggregates the ``disturbance`` column across episodes by taking the
    element-wise maximum after NaN-padding all episodes to the longest
    length, then thresholding ``> 0``.  Returns ``None`` when no episode
    exposes a ``disturbance`` column.  The result has length equal to the
    longest episode so it aligns with the ``steps`` array returned by
    :func:`_aligned_stats`.
    """
    series = [df["disturbance"].reset_index(drop=True)
              for df in dfs if "disturbance" in df.columns]
    if not series:
        return None
    maxlen = max(len(s) for s in series)
    mat = np.full((maxlen, len(series)), np.nan, dtype=float)
    for j, s in enumerate(series):
        vals = s.to_numpy(dtype=float)
        mat[:len(vals), j] = vals
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore", all="ignore"):
            agg = np.nanmax(mat, axis=1)
    mask = np.where(np.isfinite(agg), agg > 0.0, False)
    return mask.astype(bool)


def _shade_disturbance(ax, mask: Optional[np.ndarray],
                       color: str = DISTURBANCE_COLOR,
                       alpha: float = DISTURBANCE_ALPHA,
                       label: str = "Disturbance window") -> None:
    """Shade contiguous runs where ``mask`` is True using ``axvspan``.

    The first shaded run carries the legend label; subsequent runs are
    drawn unlabelled so the legend stays clean.  No-op if ``mask`` is
    ``None`` or contains no True entries.
    """
    if mask is None or len(mask) == 0:
        return
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    # Identify contiguous runs by padding with False on both ends and
    # taking the diff.  ``starts[i]`` is the index of the first True in
    # run ``i``; ``ends[i]`` is the index of the first False after it.
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    labelled = False
    for s, e in zip(starts, ends):
        # Each integer step k occupies the half-open x-interval
        # [k - 0.5, k + 0.5); a run [s, e-1] therefore spans
        # [s - 0.5, (e - 1) + 0.5 + 1) == [s - 0.5, e + 0.5).
        ax.axvspan(s - 0.5, e + 0.5, color=color, alpha=alpha,
                   linewidth=0,
                   label=(label if not labelled else None))
        labelled = True


def _episode_metrics(df: pd.DataFrame) -> dict:
    """Compute the four headline metrics for one episode DataFrame.

    Also derives ``episode_length`` from the ``terminated`` / ``truncated``
    flags when available (used for episode-length stats in the run
    summary); otherwise falls back to ``len(df)``.
    """
    m: Dict[str, float] = {
        "total_reward": float(df["reward"].sum())
        if "reward" in df.columns else 0.0,
        "mean_Tc_error": float("nan"),
        "mean_SpO2_error": float("nan"),
        "num_safety_violations": 0,
        "episode_length": int(len(df)),
    }
    if "Tc" in df.columns:
        m["mean_Tc_error"] = float(
            np.nanmean(np.abs(df["Tc"].astype(float).values - TARGET_TC)))
    if "SpO2" in df.columns:
        m["mean_SpO2_error"] = float(
            np.nanmean(np.abs(
                df["SpO2"].astype(float).values - TARGET_SPO2)))
    if "safe" in df.columns:
        m["num_safety_violations"] = int((~df["safe"].astype(bool)).sum())
    elif "Tc" in df.columns and "SpO2" in df.columns:
        tc = df["Tc"].astype(float).values
        spo2 = df["SpO2"].astype(float).values
        violations = (
            (tc < TC_SAFETY_LOWER) | (tc > TC_SAFETY_UPPER)
            | (spo2 < SPO2_SAFETY_LOWER) | (spo2 > SPO2_SAFETY_UPPER)
        )
        m["num_safety_violations"] = int(np.sum(violations))

    # Episode length: count rows up to (and including) the first
    # terminated/truncated flag, or fall back to the full row count.
    if "terminated" in df.columns or "truncated" in df.columns:
        term = df.get("terminated", pd.Series([False] * len(df))).astype(bool)
        trunc = df.get("truncated", pd.Series([False] * len(df))).astype(bool)
        end_idx = (term | trunc).values
        if end_idx.any():
            m["episode_length"] = int(np.argmax(end_idx) + 1)
    return m


def compute_scenario_metrics(dfs: Sequence[pd.DataFrame]) -> dict:
    """Aggregate the four headline metrics (+ episode length) across episodes."""
    per_ep = [_episode_metrics(df) for df in dfs]
    keys = ["total_reward", "mean_Tc_error", "mean_SpO2_error",
            "num_safety_violations", "episode_length"]
    out: Dict[str, float] = {"n_episodes": float(len(per_ep))}
    for k in keys:
        vals = np.array([m[k] for m in per_ep], dtype=float)
        out[k + "_mean"] = float(np.nanmean(vals)) if vals.size else 0.0
        out[k + "_std"] = (float(np.nanstd(vals, ddof=0))
                           if vals.size > 1 else 0.0)
    return out


def load_summary_csv(results_dir: Path) -> Optional[Dict[int, dict]]:
    """Read ``summary.csv`` (written by run_simulation.py) and aggregate per
    scenario.  Returns ``{scenario: {metric_mean, metric_std, n_episodes}}``
    or ``None`` if the file is absent / unreadable / lacks the scenario col.
    """
    path = results_dir / "summary.csv"
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover -- defensive
        print(f"[WARN] Failed to read {path}: {exc} -- will recompute "
              f"metrics from raw trajectories.", file=sys.stderr)
        return None
    if "scenario" not in df.columns:
        return None
    keys = ["total_reward", "mean_Tc_error", "mean_SpO2_error",
            "num_safety_violations"]
    out: Dict[int, dict] = {}
    for s, g in df.groupby("scenario"):
        m: Dict[str, float] = {"n_episodes": float(len(g))}
        for k in keys:
            if k in g.columns:
                vals = g[k].astype(float).to_numpy()
                m[k + "_mean"] = (float(np.nanmean(vals))
                                  if vals.size else 0.0)
                m[k + "_std"] = (float(np.nanstd(vals, ddof=0))
                                 if vals.size > 1 else 0.0)
            else:
                m[k + "_mean"], m[k + "_std"] = float("nan"), 0.0
        out[int(s)] = m
    return out


# --------------------------------------------------------------------------- #
#  Plot saving
# --------------------------------------------------------------------------- #
def _save(fig, out_path: Path, rect: Optional[Sequence[float]] = None) -> None:
    """Apply ``tight_layout`` (optionally with a ``rect`` to leave room for
    a suptitle) and write the figure at 150 dpi.

    Per project convention ``bbox_inches='tight'`` is intentionally NOT
    used with ``tight_layout``; standalone plots use ``tight_layout``
    alone.
    """
    if rect is not None:
        fig.tight_layout(rect=rect)
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Per-scenario plots
# --------------------------------------------------------------------------- #
def plot_temperature_trajectory(dfs: Sequence[pd.DataFrame], scenario: int,
                                out_path: Path) -> None:
    """Mean Tc, Tsk, Ta, Tamb (+/-1 sigma) with target & safety bounds and
    disturbance-window shading."""
    fig, ax = plt.subplots(figsize=(9, 5))

    # Shade disturbance FIRST so the data lines render on top.
    mask = _disturbance_mask(dfs)
    _shade_disturbance(ax, mask)

    for col, color in TEMP_COLORS.items():
        steps, mean, std = _aligned_stats(dfs, col)
        _band(ax, steps, mean, std, color, label=col)

    ax.axhline(TARGET_TC, color="black", linestyle="--", linewidth=1.2,
               label=f"Target Tc = {TARGET_TC:.1f} °C")
    ax.axhline(TC_SAFETY_LOWER, color="red", linestyle=":", linewidth=1.0,
               alpha=0.75,
               label=f"Safety lower = {TC_SAFETY_LOWER:.1f} °C")
    ax.axhline(TC_SAFETY_UPPER, color="red", linestyle=":", linewidth=1.0,
               alpha=0.75,
               label=f"Safety upper = {TC_SAFETY_UPPER:.1f} °C")

    n = len(dfs)
    suffix = f"  [mean of {n} episode(s) ±1σ]" if n > 1 else ""
    ax.set_title(f"Temperature Trajectory — Scenario {scenario}: "
                 f"{SCENARIO_NAMES.get(scenario, '')}{suffix}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    _save(fig, out_path)


def plot_oxygen_trajectory(dfs: Sequence[pd.DataFrame], scenario: int,
                           out_path: Path) -> None:
    """Mean SpO2 (top) and C_O2 (bottom) (+/-1 sigma) with target band &
    safety bounds and disturbance-window shading."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    mask = _disturbance_mask(dfs)
    _shade_disturbance(ax1, mask)
    _shade_disturbance(ax2, mask)

    # ---- SpO2 panel (top) ---- #
    steps, mean, std = _aligned_stats(dfs, "SpO2")
    _band(ax1, steps, mean, std, O2_COLORS["SpO2"], label="SpO2")
    ax1.axhline(SPO2_TARGET_LOWER, color="green", linestyle="--",
                linewidth=1.0, alpha=0.8,
                label=f"Target band = [{SPO2_TARGET_LOWER:.0f}, "
                      f"{SPO2_TARGET_UPPER:.0f}] %")
    ax1.axhline(SPO2_TARGET_UPPER, color="green", linestyle="--",
                linewidth=1.0, alpha=0.8)
    ax1.axhline(SPO2_SAFETY_LOWER, color="red", linestyle=":", linewidth=1.0,
                alpha=0.75,
                label=f"Safety lower = {SPO2_SAFETY_LOWER:.0f} %")
    ax1.axhline(SPO2_SAFETY_UPPER, color="red", linestyle=":", linewidth=1.0,
                alpha=0.75,
                label=f"Safety upper = {SPO2_SAFETY_UPPER:.0f} %")
    ax1.set_ylabel("SpO2 (%)")
    n = len(dfs)
    suffix = f"  [mean of {n} episode(s) ±1σ]" if n > 1 else ""
    ax1.set_title(f"Oxygen Trajectory — Scenario {scenario}: "
                  f"{SCENARIO_NAMES.get(scenario, '')}{suffix}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8, ncol=2)

    # ---- C_O2 panel (bottom) ---- #
    steps, mean, std = _aligned_stats(dfs, "C_O2")
    _band(ax2, steps, mean, std, O2_COLORS["C_O2"], label="C_O2")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("C_O2 (inspired %)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)
    _save(fig, out_path)


def plot_reward_trajectory(dfs: Sequence[pd.DataFrame], scenario: int,
                           out_path: Path) -> None:
    """Per-step reward (top) and cumulative reward (bottom), mean +/- 1
    sigma, with disturbance-window shading."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    mask = _disturbance_mask(dfs)
    _shade_disturbance(ax1, mask)
    _shade_disturbance(ax2, mask)

    steps, mean, std = _aligned_stats(dfs, "reward")
    _band(ax1, steps, mean, std, COLOR_CYCLE[2], label="Per-step reward")
    if mean is not None:
        ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("Reward")
    n = len(dfs)
    suffix = f"  [mean of {n} episode(s) ±1σ]" if n > 1 else ""
    ax1.set_title(f"Reward Trajectory — Scenario {scenario}: "
                  f"{SCENARIO_NAMES.get(scenario, '')}{suffix}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    steps, mean, std = _aligned_stats(dfs, "cumulative_reward")
    _band(ax2, steps, mean, std, COLOR_CYCLE[0], label="Cumulative reward")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Cumulative Reward")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)
    _save(fig, out_path)


def plot_control_actions(dfs: Sequence[pd.DataFrame], scenario: int,
                         out_path: Path) -> None:
    """Heater level ``u_heater`` (top) and absolute FiO2 command
    ``fio2_cmd`` (bottom), mean +/- 1 sigma, with disturbance shading."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    mask = _disturbance_mask(dfs)
    _shade_disturbance(ax1, mask)
    _shade_disturbance(ax2, mask)

    # ---- Heater level (top) ---- #
    steps, mean, std = _aligned_stats(dfs, "u_heater")
    _band(ax1, steps, mean, std, COLOR_CYCLE[3], label="Heater level (u_heater)")
    ax1.set_ylabel("Heater level (0–1)")
    ax1.set_ylim(-0.05, 1.10)
    n = len(dfs)
    suffix = f"  [mean of {n} episode(s) ±1σ]" if n > 1 else ""
    ax1.set_title(f"Control Actions — Scenario {scenario}: "
                  f"{SCENARIO_NAMES.get(scenario, '')}{suffix}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    # ---- FiO2 command (bottom) -- absolute fraction, 0.21-0.60. ---- #
    steps, mean, std = _aligned_stats(dfs, "fio2_cmd")
    _band(ax2, steps, mean, std, COLOR_CYCLE[4],
          label="FiO2 command (fio2_cmd)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("FiO2 (fraction)")
    # Pad the y-limits around the physiological range.
    ax2.set_ylim(0.20, 0.62)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
#  Cross-scenario summaries
# --------------------------------------------------------------------------- #
def plot_summary_bar(metrics_by_scenario: Dict[int, dict],
                     out_path: Path) -> None:
    """2x2 bar chart comparing scenarios on the four headline metrics.

    Error bars are +/-1 sigma across episodes (zero when only one
    episode is available).
    """
    scenarios = sorted(metrics_by_scenario.keys())
    if not scenarios:
        print("[WARN] No scenario metrics available for summary bar chart "
              "-- skipping.", file=sys.stderr)
        return

    labels = [f"S{s}" for s in scenarios]
    x = np.arange(len(scenarios))
    keys = ["total_reward", "mean_Tc_error", "mean_SpO2_error",
            "num_safety_violations"]
    titles = ["Total Episode Reward (per-episode mean)",
              "Mean |Tc − 37 °C|  (°C)",
              "Mean |SpO2 − 92 %|  (%)",
              "Safety Violations (#/episode)"]
    colors = [COLOR_CYCLE[0], COLOR_CYCLE[1], COLOR_CYCLE[2], COLOR_CYCLE[3]]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, key, title, color in zip(axes.flat, keys, titles, colors):
        means = [metrics_by_scenario[s].get(key + "_mean", 0.0)
                 for s in scenarios]
        stds = [metrics_by_scenario[s].get(key + "_std", 0.0)
                for s in scenarios]
        # Suppress error bars that are NaN/inf so capsize rendering is clean.
        yerr = [v if np.isfinite(v) and v > 0 else 0.0 for v in stds]
        ax.bar(x, means, color=color, edgecolor="black", linewidth=0.5,
               width=0.6,
               yerr=yerr if any(yerr) else None, capsize=3)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, axis="y", alpha=0.3)
        # Annotate each bar with its mean value (offset above the bar).
        finite_means = [m for m in means if np.isfinite(m)]
        ymax = max(finite_means, default=1.0)
        ymin = min(finite_means, default=0.0)
        span = (ymax - ymin) if np.isfinite(ymax) and np.isfinite(ymin) else 1.0
        offset = 0.02 * (abs(span) if span else 1.0)
        for xi, v in zip(x, means):
            if not np.isfinite(v):
                continue
            va = "bottom" if v >= 0 else "top"
            yy = v + offset if v >= 0 else v - offset
            ax.text(xi, yy, f"{v:.2f}", ha="center", va=va, fontsize=8)
        # Guarantee headroom so annotations / error bars are not clipped.
        hi = max(ymax, 0.0)
        lo = min(ymin, 0.0)
        pad = 0.18 * (abs(hi - lo) if hi != lo else max(abs(hi), 1.0))
        ax.set_ylim(lo - pad, hi + pad)

    fig.suptitle("Scenario Comparison — Baseline Performance Summary",
                 fontsize=13, fontweight="bold")
    _save(fig, out_path, rect=[0, 0.03, 1, 0.95])


def plot_scenario_comparison_temperature(
        traj_by_scenario: Dict[int, List[pd.DataFrame]],
        out_path: Path) -> None:
    """Overlay mean Tc trajectories for all scenarios on a single axis."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, s in enumerate(sorted(traj_by_scenario.keys())):
        steps, mean, _ = _aligned_stats(traj_by_scenario[s], "Tc")
        if mean is None:
            continue
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        ax.plot(steps, mean, color=color, linewidth=1.5,
                label=f"S{s}: {SCENARIO_NAMES.get(s, '')}")
    ax.axhline(TARGET_TC, color="black", linestyle="--", linewidth=1.2,
               label=f"Target = {TARGET_TC:.1f} °C")
    ax.axhline(TC_SAFETY_LOWER, color="red", linestyle=":", linewidth=1.0,
               alpha=0.6, label=f"Safety lower = {TC_SAFETY_LOWER:.1f} °C")
    ax.axhline(TC_SAFETY_UPPER, color="red", linestyle=":", linewidth=1.0,
               alpha=0.6, label=f"Safety upper = {TC_SAFETY_UPPER:.1f} °C")
    ax.set_title("Core Temperature (Tc) — All Scenarios (per-scenario mean)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Tc (°C)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _save(fig, out_path)


def plot_scenario_comparison_spo2(
        traj_by_scenario: Dict[int, List[pd.DataFrame]],
        out_path: Path) -> None:
    """Overlay mean SpO2 trajectories for all scenarios on a single axis."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, s in enumerate(sorted(traj_by_scenario.keys())):
        steps, mean, _ = _aligned_stats(traj_by_scenario[s], "SpO2")
        if mean is None:
            continue
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        ax.plot(steps, mean, color=color, linewidth=1.5,
                label=f"S{s}: {SCENARIO_NAMES.get(s, '')}")
    ax.axhline(SPO2_TARGET_LOWER, color="green", linestyle="--",
               linewidth=1.0, alpha=0.8,
               label=f"Target band = [{SPO2_TARGET_LOWER:.0f}, "
                     f"{SPO2_TARGET_UPPER:.0f}] %")
    ax.axhline(SPO2_TARGET_UPPER, color="green", linestyle="--",
               linewidth=1.0, alpha=0.8)
    ax.axhline(SPO2_SAFETY_LOWER, color="red", linestyle=":", linewidth=1.0,
               alpha=0.6, label=f"Safety lower = {SPO2_SAFETY_LOWER:.0f} %")
    ax.axhline(SPO2_SAFETY_UPPER, color="red", linestyle=":", linewidth=1.0,
               alpha=0.6, label=f"Safety upper = {SPO2_SAFETY_UPPER:.0f} %")
    ax.set_title("Oxygen Saturation (SpO2) — All Scenarios (per-scenario mean)")
    ax.set_xlabel("Step")
    ax.set_ylabel("SpO2 (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
#  CLI entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        prog="generate_plots.py",
        description=(
            "Member 1 -- generate baseline-performance plots from the "
            "per-scenario trajectory CSVs produced by run_simulation.py "
            "(NEW 11-dim observation schema with phase + disturbance "
            "columns)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing the trajectory CSVs "
             "(trajectory_scenario{S}[_ep{E}].csv) and optionally "
             "summary.csv.",
    )
    p.add_argument(
        "--plots-dir",
        default="plots",
        help="Directory to write PNG plots into. Created if missing.",
    )
    p.add_argument(
        "--scenarios",
        default="1,2,3,4,5",
        help="Comma-separated scenario IDs to process.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point: load trajectories, render all 7 plot families."""
    args = parse_args(argv)

    results_dir = Path(args.results_dir).resolve()
    plots_dir = Path(args.plots_dir).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        scenarios = [int(tok.strip()) for tok in args.scenarios.split(",")
                     if tok.strip()]
    except ValueError:
        print(f"[ERROR] Could not parse --scenarios {args.scenarios!r}; "
              f"expected comma-separated integers (e.g. 1,2,3,4,5).",
              file=sys.stderr)
        return 2

    print(f"[INFO] Results dir: {results_dir}")
    print(f"[INFO] Plots dir:   {plots_dir}")
    print(f"[INFO] Scenarios:   {scenarios}")

    if not results_dir.exists():
        print(f"[WARN] Results directory does not exist: {results_dir} -- "
              f"nothing to plot.", file=sys.stderr)
        return 0

    # Prefer the authoritative summary.csv for the bar chart; fall back to
    # recomputing per-episode metrics from the raw trajectories.
    summary_from_csv = load_summary_csv(results_dir)

    generated: List[str] = []
    traj_by_scenario: Dict[int, List[pd.DataFrame]] = {}
    metrics_by_scenario: Dict[int, dict] = {}

    for s in scenarios:
        dfs = load_scenario_episodes(results_dir, s)
        if not dfs:
            continue
        traj_by_scenario[s] = dfs
        # Always compute metrics from the raw trajectories so we have
        # ``episode_length_*`` (not in summary.csv); when summary.csv is
        # available override the four headline metrics with its values
        # (authoritative source).
        computed = compute_scenario_metrics(dfs)
        if summary_from_csv and s in summary_from_csv:
            summary = summary_from_csv[s]
            for k in ("total_reward", "mean_Tc_error",
                      "mean_SpO2_error", "num_safety_violations"):
                computed[k + "_mean"] = summary.get(k + "_mean",
                                                    computed[k + "_mean"])
                computed[k + "_std"] = summary.get(k + "_std",
                                                   computed[k + "_std"])
        metrics_by_scenario[s] = computed

        for fname, fn in [
            (f"temperature_trajectory_scenario{s}.png",
             plot_temperature_trajectory),
            (f"oxygen_trajectory_scenario{s}.png",
             plot_oxygen_trajectory),
            (f"reward_trajectory_scenario{s}.png",
             plot_reward_trajectory),
            (f"control_actions_scenario{s}.png",
             plot_control_actions),
        ]:
            out_path = plots_dir / fname
            fn(dfs, s, out_path)
            generated.append(fname)

        m = metrics_by_scenario[s]
        # Episode-length stats come from the per-episode metrics (derived
        # from terminated/truncated flags when available).
        n_eps = int(m.get("n_episodes", 0))
        ep_len_mean = m.get("episode_length_mean", float("nan"))
        ep_len_std = m.get("episode_length_std", 0.0)
        print(f"[OK] Scenario {s}: {n_eps} episode(s) | "
              f"total_reward={m['total_reward_mean']:.2f}"
              f" ± {m['total_reward_std']:.2f} | "
              f"Tc_err={m['mean_Tc_error_mean']:.4f} | "
              f"SpO2_err={m['mean_SpO2_error_mean']:.4f} | "
              f"viol/ep={m['num_safety_violations_mean']:.1f}"
              f" ± {m['num_safety_violations_std']:.1f} | "
              f"ep_len={ep_len_mean:.1f} ± {ep_len_std:.1f}")

    if traj_by_scenario:
        for fname, fn in [
            ("summary_bar.png", plot_summary_bar),
            ("scenario_comparison_temperature.png",
             plot_scenario_comparison_temperature),
            ("scenario_comparison_spo2.png",
             plot_scenario_comparison_spo2),
        ]:
            out_path = plots_dir / fname
            if fname == "summary_bar.png":
                fn(metrics_by_scenario, out_path)
            else:
                fn(traj_by_scenario, out_path)
            generated.append(fname)
    else:
        print("[WARN] No scenario trajectories were loaded -- the cross-"
              "scenario summary plots were not generated.", file=sys.stderr)

    print(f"\n[DONE] Generated {len(generated)} plot(s) in {plots_dir}:")
    for name in generated:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
