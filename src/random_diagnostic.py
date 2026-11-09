#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 random_diagnostic.py -- V5-04 exploration diagnostic (Member 1)
------------------------------------------------------------------------------
Project : Project 04 -- Neonatal Incubator Regulation (RL + XAI)
          Member 1 deliverable: Simulation Environment, Dataset & MDP.
Module  : random_diagnostic.py
Purpose : Run a uniform-random policy across all 5 incubator scenarios and
          report episode-length / safety / termination statistics. This
          addresses V5-04: "Member 1: supply an easy-to-run random-policy
          diagnostic script and the episode-length/safety statistics."

V5-04 DIAGNOSTIC
----------------
V5-04 diagnostic: runs a uniform-random policy to characterize early-
termination behavior under random exploration. This is a training-risk
diagnostic, NOT a simulator bug -- random actions are expected to be
unsafe. Member 2 should use this to understand the exploration challenge
before DDQN training.

A uniform-random policy over Discrete(25) (5 heater levels x 5 FiO2
levels) will, on average, drastically mistune both actuators. Without
a sensible controller the core temperature (Tc) drifts outside
[TC_SAFE_LOW, TC_SAFE_HIGH] = [35.5, 38.0] deg C, or the arterial
oxygen saturation (SpO2) leaves [SPO2_SAFE_LOW, SPO2_SAFE_HIGH] =
[85.0, 96.0] %, and the environment terminates the episode with a
safety penalty. The episode-length distribution under random play is
therefore a direct measure of how "forgiving" each scenario's initial
state and disturbance profile are -- and consequently how hard the
exploration problem is that the DDQN agent (Member 2) must solve.

WHAT THIS SCRIPT MEASURES
-------------------------
For every (scenario, episode) pair it records:

* ``scenario``            : 1..5 (matches environment.scenario)
* ``episode``             : 0-based episode index within the scenario
* ``seed``                : episode_seed = base_seed + scenario*1000 + episode
                            (same convention as run_simulation.py)
* ``total_reward``        : cumulative reward over the episode
* ``episode_length``      : number of control steps actually executed
                            (terminated/truncated step included)
* ``terminated``          : True if the env flagged a hard safety breach
* ``truncated``           : True if the env hit max_steps (200 steps)
* ``termination_cause``   : derived from the info dict's safety flags
                            at the final step -- one of
                            {hypothermia, hyperthermia, hypoxia,
                            hyperoxia, max_steps, safe, unknown}.
                            Multiple flags may be joined with '+'.
* ``final_Tc``            : core temperature at the terminal step (deg C)
* ``final_SpO2``          : arterial oxygen saturation at the terminal step (%)

OUTPUTS
-------
* ``results/random_diagnostic.csv``        : one row per (scenario, episode)
* ``plots/random_diagnostic_episode_length.png`` :
    bar chart of mean episode length per scenario (error bars = +/-1 std)
* a human-readable summary table on stdout giving, per scenario, the
  mean episode length, mean reward, % terminated early, and most
  common termination cause.

USAGE
-----
Run from the Member 1 root directory::

    python src/random_diagnostic.py --episodes 20 --seed 42
    python src/random_diagnostic.py --episodes 50 --seed 7 \
        --output-dir results --plots-dir plots

EXIT CODES
----------
0 : all episodes ran and outputs were written successfully
1 : unexpected error during the run (message on stderr)
==============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  Make sibling modules (environment.py) importable regardless of the         #
#  caller's working directory.                                                #
# --------------------------------------------------------------------------- #
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from environment import NeonatalIncubatorEnv  # noqa: E402


# --------------------------------------------------------------------------- #
#  Constants                                                                  #
# --------------------------------------------------------------------------- #
# The five canonical clinical scenarios defined in environment.py / config.yaml.
DEFAULT_SCENARIOS: List[int] = [1, 2, 3, 4, 5]

# Per-episode diagnostic CSV columns (frozen contract for Members 2 & 3).
DIAGNOSTIC_COLUMNS: List[str] = [
    "scenario",
    "episode",
    "seed",
    "total_reward",
    "episode_length",
    "terminated",
    "truncated",
    "termination_cause",
    "final_Tc",
    "final_SpO2",
]

# Priority order in which safety flags are reported when multiple are
# simultaneously True at the terminal step. The order mirrors clinical
# severity (hypothermia is the dominant failure mode for random play in
# scenarios 1/2/3/4; hypoxia dominates in scenario 5 because the FiO2
# actuator is randomly mistuned during the desaturation window).
_SAFETY_FLAG_PRIORITY: List[str] = [
    "hypothermia",
    "hyperthermia",
    "hypoxia",
    "hyperoxia",
]


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
def derive_termination_cause(
    info: Dict[str, Any], terminated: bool, truncated: bool
) -> str:
    """Derive a human-readable termination cause from the final ``info``.

    The environment's ``info`` dict exposes the boolean safety flags
    ``hypothermia`` / ``hyperthermia`` / ``hypoxia`` / ``hyperoxia`` /
    ``safe`` at every step (see ``NeonatalIncubatorEnv._build_info``).
    At the terminal step these reflect the post-step state that
    triggered termination.

    Rules
    -----
    * If ``truncated`` is True -> ``"max_steps"`` (the env reached
      ``current_step >= max_steps`` without breaching any hard limit).
    * Else if any of the four safety flags is True -> the matching
      cause(s), joined with ``+`` if multiple are simultaneously True
      (priority order: hypothermia > hyperthermia > hypoxia > hyperoxia).
    * Else if ``info['safe']`` is True -> ``"safe"`` (terminated was set
      but no safety flag flipped -- should not happen for the current
      env, but is recorded defensively so the CSV is never empty).
    * Otherwise -> ``"unknown"``.

    Parameters
    ----------
    info : dict
        The info dict returned by the env's terminal ``step()`` call.
    terminated : bool
        ``terminated`` flag returned by the env at the terminal step.
    truncated : bool
        ``truncated`` flag returned by the env at the terminal step.

    Returns
    -------
    str
        One of {hypothermia, hyperthermia, hypoxia, hyperoxia, max_steps,
        safe, unknown} (possibly multiple safety causes joined by '+').
    """
    info = info or {}

    # Truncation takes precedence: the episode ran to max_steps without
    # breaching any hard safety limit.
    if truncated:
        return "max_steps"

    # If terminated=True the env flagged a hard safety breach; identify which.
    if terminated:
        active = [
            cause
            for cause in _SAFETY_FLAG_PRIORITY
            if bool(info.get(cause, False))
        ]
        if active:
            return "+".join(active)
        # Defensive: terminated was set but no specific flag is True.
        if bool(info.get("safe", False)):
            return "safe"
        return "unknown"

    # Episode ended without terminated or truncated (shouldn't happen for
    # this env, but record defensively).
    if bool(info.get("safe", False)):
        return "safe"
    return "unknown"


def run_random_episode(
    env: NeonatalIncubatorEnv,
    scenario: int,
    episode: int,
    base_seed: int,
) -> Dict[str, Any]:
    """Run a single episode under a uniform-random policy.

    Parameters
    ----------
    env : NeonatalIncubatorEnv
        Environment instance (will be ``reset(seed=...)`` internally).
    scenario : int
        Scenario id (1..5); used for per-(scenario, episode) seeding.
    episode : int
        0-based episode index within the scenario.
    base_seed : int
        Base RNG seed. The per-episode seed is
        ``base_seed + scenario*1000 + episode`` -- identical to the
        convention used by ``run_simulation.py``.

    Returns
    -------
    dict
        One row of the diagnostic CSV with keys from
        :data:`DIAGNOSTIC_COLUMNS`.
    """
    # Per-(scenario, episode) reproducible seeding. Matches run_simulation.py.
    episode_seed: int = int(base_seed) + int(scenario) * 1000 + int(episode)

    # Seed numpy's GLOBAL RNG so np.random.randint draws are reproducible.
    # NOTE: the env uses its own private np.random.default_rng(seed) via
    # reset(seed=...), so this does not perturb the env's stochastic
    # disturbances -- the two RNG streams are independent.
    np.random.seed(episode_seed)

    obs, info = env.reset(seed=episode_seed)

    total_reward: float = 0.0
    steps: int = 0
    terminated: bool = False
    truncated: bool = False
    last_info: Dict[str, Any] = info

    # Uniform-random policy: each action is drawn uniformly from
    # Discrete(action_space.n). Episode ends on terminated OR truncated.
    while not (terminated or truncated):
        action = int(np.random.randint(env.action_space.n))
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += float(reward)
        last_info = info

    term_cause = derive_termination_cause(
        last_info, terminated=terminated, truncated=truncated
    )
    final_tc = float(last_info.get("Tc", float("nan")))
    final_spo2 = float(last_info.get("SpO2", float("nan")))

    return {
        "scenario": int(scenario),
        "episode": int(episode),
        "seed": int(episode_seed),
        "total_reward": float(total_reward),
        "episode_length": int(steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_cause": term_cause,
        "final_Tc": final_tc,
        "final_SpO2": final_spo2,
    }


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-episode diagnostics into a per-scenario summary.

    Columns
    -------
    scenario : int
    mean_episode_length : float
    std_episode_length : float
    mean_reward : float
    pct_terminated_early : float
        Percentage of episodes in this scenario that terminated early
        (i.e. ``terminated == True``); truncated episodes (max_steps)
        are NOT counted as "early".
    most_common_cause : str
        Most frequent termination_cause across the scenario's episodes
        (ties broken by first-occurrence order).
    n_episodes : int
    """
    rows: List[Dict[str, Any]] = []
    for scenario, grp in df.groupby("scenario", sort=True):
        n = int(len(grp))
        mean_len = float(grp["episode_length"].mean())
        std_len = float(grp["episode_length"].std(ddof=0))
        mean_rew = float(grp["total_reward"].mean())
        pct_early = float(
            (grp["terminated"].astype(bool).sum() / n) * 100.0
        ) if n > 0 else 0.0
        # Most common termination cause (mode). value_counts returns
        # counts in descending order; .index[0] is the most common.
        if n > 0:
            cause_counts = grp["termination_cause"].value_counts()
            if len(cause_counts) > 0:
                most_common = str(cause_counts.index[0])
            else:
                most_common = "n/a"
        else:
            most_common = "n/a"
        rows.append({
            "scenario": int(scenario),
            "mean_episode_length": mean_len,
            "std_episode_length": std_len,
            "mean_reward": mean_rew,
            "pct_terminated_early": pct_early,
            "most_common_cause": most_common,
            "n_episodes": n,
        })
    return pd.DataFrame(rows)


def render_episode_length_plot(
    df: pd.DataFrame, plot_path: Path
) -> None:
    """Render the per-scenario mean episode-length bar chart.

    Bars show mean episode length per scenario; error bars are +/-1
    standard deviation (population std, ddof=0) across the episodes.
    """
    import matplotlib
    matplotlib.use("Agg")  # noqa: E402  must precede pyplot import
    import matplotlib.pyplot as plt  # noqa: E402

    summary = build_summary_table(df).sort_values("scenario")

    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = summary["scenario"].astype(int).tolist()
    means = summary["mean_episode_length"].astype(float).values
    stds = summary["std_episode_length"].astype(float).values

    bars = ax.bar(
        scenarios,
        means,
        yerr=stds,
        capsize=6,
        color="#4c72b0",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
        label="mean +/- 1 std",
    )
    # Annotate each bar with the mean value.
    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{mean_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Scenario", fontsize=11)
    ax.set_ylabel("Episode Length (control steps)", fontsize=11)
    ax.set_title(
        "V5-04 Random-Policy Diagnostic: Mean Episode Length per Scenario\n"
        "(uniform-random actions; error bars = +/-1 std)",
        fontsize=11,
    )
    ax.set_xticks(scenarios)
    ax.set_xticklabels(
        [f"S{s}\n(n={int(n)})" for s, n in
         zip(scenarios, summary["n_episodes"].astype(int).tolist())]
    )
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="random_diagnostic.py",
        description=(
            "V5-04 exploration diagnostic: run a uniform-random policy "
            "across all 5 incubator scenarios and report episode-length / "
            "safety / termination statistics."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes to run per scenario (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Base RNG seed. Per-(scenario, episode) seed is derived as "
            "seed + scenario*1000 + episode for reproducibility, "
            "matching run_simulation.py (default: 42)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help=(
            "Directory to write random_diagnostic.csv (default: results). "
            "Created if it does not exist. Interpreted relative to the "
            "caller's working directory."
        ),
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots",
        help=(
            "Directory to write random_diagnostic_episode_length.png "
            "(default: plots). Created if it does not exist."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point: run all (scenario, episode) pairs and persist results."""
    args = parse_args(argv)

    if args.episodes <= 0:
        print(
            f"[random_diagnostic] --episodes must be > 0, got "
            f"{args.episodes}",
            file=sys.stderr,
        )
        return 2

    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Member 1 -- V5-04 Random-Policy Diagnostic")
    print("Neonatal Incubator Temperature & Oxygen Regulation (DDQN + SHAP)")
    print("=" * 72)
    print(f"Scenarios       : {DEFAULT_SCENARIOS}")
    print(f"Episodes/scn    : {args.episodes}")
    print(f"Base seed       : {args.seed}")
    print(f"Seed convention : seed + scenario*1000 + episode  "
          f"(matches run_simulation.py)")
    print(f"Output dir      : {output_dir.resolve()}")
    print(f"Plots dir       : {plots_dir.resolve()}")
    print("-" * 72)
    print(
        "V5-04 diagnostic: uniform-random policy characterization of "
        "early-termination behaviour."
    )
    print(
        "This is a TRAINING-RISK diagnostic, NOT a simulator bug -- "
        "random actions are expected to be unsafe."
    )
    print("-" * 72)

    rows: List[Dict[str, Any]] = []

    for scenario in DEFAULT_SCENARIOS:
        for episode in range(args.episodes):
            # Re-instantiate the env per (scenario, episode) so the env's
            # private RNG is re-seeded -> fully reproducible episodes
            # (same pattern as run_simulation.py).
            ep_seed: int = int(args.seed) + int(scenario) * 1000 + int(episode)
            env = NeonatalIncubatorEnv(scenario=scenario, seed=ep_seed)
            try:
                row = run_random_episode(
                    env=env,
                    scenario=scenario,
                    episode=episode,
                    base_seed=args.seed,
                )
            finally:
                if hasattr(env, "close"):
                    try:
                        env.close()
                    except Exception:  # pragma: no cover - defensive
                        pass

            rows.append(row)
            print(
                f"  [scn {scenario} | ep {episode:2d} | seed {row['seed']}] "
                f"len={row['episode_length']:3d}  "
                f"reward={row['total_reward']:10.2f}  "
                f"term={str(row['terminated']):5s}  "
                f"trunc={str(row['truncated']):5s}  "
                f"cause={row['termination_cause']:<22s}  "
                f"Tc={row['final_Tc']:.2f}  SpO2={row['final_SpO2']:.1f}"
            )

    df = pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)
    csv_path = output_dir / "random_diagnostic.csv"
    df.to_csv(csv_path, index=False)

    # ---- Summary table -------------------------------------------------- #
    summary_df = build_summary_table(df)

    print("-" * 72)
    print("Per-scenario summary (uniform-random policy):")
    print("-" * 72)
    # Render with a clean format for stdout.
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 120,
        "display.float_format", lambda v: f"{v:8.3f}",
    ):
        print(summary_df.to_string(index=False))
    print("-" * 72)
    # Pretty one-line recap per scenario for quick reading.
    for _, r in summary_df.iterrows():
        print(
            f"  Scenario {int(r['scenario'])}: "
            f"mean_len={r['mean_episode_length']:5.2f} "
            f"(+/-{r['std_episode_length']:4.2f})  "
            f"mean_reward={r['mean_reward']:8.2f}  "
            f"%early_term={r['pct_terminated_early']:5.1f}%  "
            f"most_common_cause={r['most_common_cause']}"
        )
    print("-" * 72)

    # ---- Plot ----------------------------------------------------------- #
    plot_path = plots_dir / "random_diagnostic_episode_length.png"
    try:
        render_episode_length_plot(df, plot_path)
        print(f"Wrote plot -> {plot_path.resolve()}")
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"[random_diagnostic] WARNING: could not render plot "
            f"{plot_path}: {exc}",
            file=sys.stderr,
        )

    print(
        f"Wrote {len(df)} per-episode rows -> "
        f"{csv_path.resolve()}"
    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
