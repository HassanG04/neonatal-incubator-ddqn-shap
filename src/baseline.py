#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 baseline.py -- Rule-based (PID + bang-bang) baseline controller
------------------------------------------------------------------------------
Project : Member 1 deliverable -- RL + XAI for Neonatal Incubator
          Temperature & Oxygen Regulation (DDQN + SHAP).
Module  : baseline.py
Purpose : A non-RL reference controller that can be compared head-to-head
          against the DDQN agent developed by Member 2.  The controller
          regulates the two controlled variables of the incubator
          environment -- core body temperature ``Tc`` and arterial oxygen
          saturation ``SpO2`` -- using two complementary strategies:

          * **Temperature channel**: a continuous PID controller (with
            anti-windup clamping) whose ``[0, 1]`` heater command is
            snapped onto the same five discrete heater levels used by
            ``NeonatalIncubatorEnv``.

          * **Oxygen channel**: a rule-based (bang-bang with dead-zone)
            FiO2 mapper that escalates the inspired O2 fraction in three
            steps as the hypoxaemia gap grows and shuts supplemental O2
            off when the neonate is already inside the target band.

          The two channel indices (``heater_idx in {0..4}`` and
          ``o2_idx in {0..4}``) are combined into the single discrete
          action ``a = heater_idx * 5 + o2_idx`` in ``[0, 24]`` so the
          baseline is action-space-compatible with the DDQN agent
          (``NeonatalIncubatorEnv.action_space == Discrete(25)``).

Gymnasium API used (5-tuple step)::

    obs, info       = env.reset(seed=42)
    obs, r, term, trunc, info = env.step(action)

Observation (11-dim, float32, feature order frozen by ``environment.OBS_NAMES``)::

    [ Tc, Tsk, Ta, C_O2, SpO2, Tamb, dTa_dt, dC_O2_dt, phase,
      disturbance, scenario_id ]
       0    1    2    3     4     5      6        7        8        9        10

>>> SIMULATION PARAMETERS, NOT CLINICAL GUIDANCE. <<<
==============================================================================
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np


__all__ = ["RuleBasedBaselineController", "HEATER_LEVELS", "FIO2_LEVELS"]


# ---------------------------------------------------------------------------
#  Actuator level tables -- mirror environment.HEATER_LEVELS / FIO2_LEVELS.
#  Imported from environment.py when available (single source of truth);
#  fall back to the literal defaults below for unit tests of the controller
#  that don't have environment.py on the import path.
# ---------------------------------------------------------------------------
# Fraction of Q_HEATER_MAX (360 W) delivered to the incubator air node.
try:  # pragma: no cover - exercised whenever environment.py is importable.
    from environment import HEATER_LEVELS as _ENV_HEATER_LEVELS
    from environment import FIO2_LEVELS as _ENV_FIO2_LEVELS
    HEATER_LEVELS: tuple[float, ...] = _ENV_HEATER_LEVELS
    FIO2_LEVELS: tuple[float, ...] = _ENV_FIO2_LEVELS
except Exception:  # noqa: BLE001 -- defensive fallback for env-less imports.
    HEATER_LEVELS = (0.0, 0.25, 0.50, 0.75, 1.0)
    FIO2_LEVELS = (0.21, 0.30, 0.40, 0.50, 0.60)
# (FIO2_LEVELS is the delivered inspired oxygen fraction; values are the
#  absolute volume percent in [0.21, 0.60], NOT normalized to [0,1].)


class RuleBasedBaselineController:
    """Rule-based baseline controller: PID heater + bang-bang FiO2.

    The controller is intentionally simple, transparent and deterministic so
    that it can serve as a reference point against which the DDQN agent can
    be benchmarked and (via SHAP) explained.

    Temperature channel (PID)
    -------------------------
    A textbook parallel-form PID controller targeting ``target_Tc`` (default
    37.0 deg C, the mid of the WHO/AAP normal band [36.5, 37.5])::

        error      = target_Tc - Tc
        integral  += error * dt            (clamped to +/- integral_clamp)
        derivative = (error - prev_error) / dt     (0 on the first call)
        output     = Kp*error + Ki*integral + Kd*derivative
        output     = clip(output, 0, 1)
        heater_idx = argmin_k |HEATER_LEVELS[k] - output|

    Anti-windup is implemented by clamping the integrator magnitude to
    ``integral_clamp`` (default 1.0) on every update, which prevents
    integrator blow-up during sustained saturation.

    Oxygen channel (bang-bang with dead-zone)
    -----------------------------------------
    A four-region rule-based mapper targeting ``target_SpO2`` (default 93%,
    the mid of the NICE/AAP post-stabilization band [91, 95])::

        SpO2 < 89           -> FiO2 = 0.40   (o2_idx = 2)  -> SpO2_eq ~ 94.5
        89 <= SpO2 < 91     -> FiO2 = 0.30   (o2_idx = 1)  -> SpO2_eq ~ 93.5
        91 <= SpO2 <= 95    -> FiO2 = 0.21   (o2_idx = 0)  -> SpO2_eq ~ 92.0
        SpO2 > 95           -> FiO2 = 0.21   (o2_idx = 0)  (room air, reduce)

    The [91, 95] region is the explicit dead-zone: when the neonate is
    already inside the target band, supplemental O2 is shut off to save
    energy and avoid hyperoxia.

    Action encoding
    ---------------
    The two channel indices are combined into a single discrete action::

        action = heater_idx * 5 + o2_idx     # in [0, 24]

    This is bit-for-bit compatible with ``NeonatalIncubatorEnv.action_space``
    (``Discrete(25)``) and with ``environment.ACTION_TABLE``.

    Parameters
    ----------
    target_Tc : float, default 37.0
        Core temperature setpoint (deg C).
    target_SpO2 : float, default 93.0
        Arterial oxygen saturation setpoint (%).  Used as the design centre
        of the bang-bang dead-zone; the dead-zone itself is [91, 95].
    dt : float, default 45.0
        Control interval (s).  Must match ``NeonatalIncubatorEnv.dt`` so
        that the PID integral/derivative terms are scaled consistently.
    Kp, Ki, Kd : float
        PID gains (default 0.8, 0.05, 0.1).
    integral_clamp : float, default 1.0
        Anti-windup magnitude clamp on the PID integrator.

    Attributes
    ----------
    last_info : dict
        PID internals exposed for logging / SHAP analysis.  Keys:
        ``error``, ``integral``, ``derivative``, ``output`` (the four PID
        signals requested by the spec), plus ``heater_idx``, ``o2_idx``,
        ``heater_level``, ``fio2_command``, ``target_Tc``, ``target_SpO2``.
    """

    # --------------------------- class-level config ----------------------- #
    # PID gains (tuned for the env's core time constant C_CORE/K_BABY = 500 s
    # and a 45 s control interval -> ~11 steps per core time constant).
    DEFAULT_KP: float = 0.8
    DEFAULT_KI: float = 0.05
    DEFAULT_KD: float = 0.1
    DEFAULT_INTEGRAL_CLAMP: float = 1.0   # |integral| <= 1.0 (anti-windup)

    # Feed-forward bias: at the setpoint (error=0) the incubator still needs
    # heat to counter ambient loss. Steady-state Q_heater = K_LOSS*(Tc-Tamb)
    # = 8.0*(37-25) = 96 W -> u_heater = 96/360 = 0.267, closest discrete
    # heater level is 0.25 (idx 1). We bias the PID to 0.25 so heater_idx
    # snaps to 1 at setpoint, maintaining Tc without overshoot. (The previous
    # value 0.5 was tuned for the legacy K_LOSS=15 -> 180 W, 0.5; with the
    # current well-insulated K_LOSS=8.0 it caused a steady ~0.2 deg C
    # overshoot above the 37.0 deg C setpoint.)
    FF_BIAS: float = 0.25

    # Default setpoints.
    DEFAULT_TARGET_TC: float = 37.0       # deg C -- mid of [36.5, 37.5]
    DEFAULT_TARGET_SPO2: float = 93.0     # %     -- mid of [91, 95]

    # Bang-bang FiO2 thresholds (the dead-zone is [SPO2_LO, SPO2_HI]).
    SPO2_SEVERE_HYPOXIA: float = 89.0     # below -> FiO2 = 0.40
    SPO2_MILD_HYPOXIA: float = 91.0       # below -> FiO2 = 0.30
    SPO2_TARGET_LOW: float = 91.0         # dead-zone lower edge
    SPO2_TARGET_HIGH: float = 95.0        # dead-zone upper edge
    SPO2_HYPEROXIA: float = 95.0          # above -> FiO2 = 0.21

    # Default control interval (must match environment.DT_STEP).
    DEFAULT_DT: float = 45.0

    # ------------------------------------------------------------------ #
    #  Construction                                                      #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        target_Tc: float | None = None,
        target_SpO2: float | None = None,
        dt: float | None = None,
        Kp: float | None = None,
        Ki: float | None = None,
        Kd: float | None = None,
        integral_clamp: float | None = None,
        ff_bias: float | None = None,
        config_path: str | None = None,
        heater_levels: Optional[Sequence[float]] = None,
        fio2_levels: Optional[Sequence[float]] = None,
    ) -> None:
        # V4-02: optionally read defaults from config.yaml (same source as env).
        if config_path is not None or any(
            v is None for v in (target_Tc, target_SpO2, dt, Kp, Ki, Kd, ff_bias,
                                integral_clamp, heater_levels, fio2_levels)
        ):
            try:
                from environment import load_config
                cfg = load_config(config_path)
            except ImportError:
                cfg = {}
        else:
            cfg = {}
        # Constructor kwargs override YAML which overrides class defaults.
        self.target_Tc = float(target_Tc if target_Tc is not None
                               else cfg.get("baseline_target_Tc", self.DEFAULT_TARGET_TC))
        self.target_SpO2 = float(target_SpO2 if target_SpO2 is not None
                                 else cfg.get("baseline_target_SpO2", self.DEFAULT_TARGET_SPO2))
        self.dt = float(dt if dt is not None else cfg.get("dt", self.DEFAULT_DT))
        self.Kp = float(Kp if Kp is not None else cfg.get("baseline_Kp", self.DEFAULT_KP))
        self.Ki = float(Ki if Ki is not None else cfg.get("baseline_Ki", self.DEFAULT_KI))
        self.Kd = float(Kd if Kd is not None else cfg.get("baseline_Kd", self.DEFAULT_KD))
        self.integral_clamp = float(
            integral_clamp if integral_clamp is not None
            else cfg.get("baseline_integral_clamp", self.DEFAULT_INTEGRAL_CLAMP))
        # V4-02: FF_BIAS from config if provided (else class default).
        self.FF_BIAS = float(ff_bias if ff_bias is not None
                             else cfg.get("baseline_ff_bias", 0.25))

        # Actuator level tables. Priority: explicit kwargs (e.g. via
        # ``from_env``) > YAML config > module defaults.
        # Fix B: grids are VALIDATED here so an invalid / mismatched grid
        # fails early with a clear ValueError instead of an IndexError later.
        self.heater_levels: tuple[float, ...] = self._validate_levels(
            heater_levels if heater_levels is not None
            else cfg.get("heater_levels", list(HEATER_LEVELS)),
            name="heater_levels", lo=0.0, hi=1.0)
        self.fio2_levels: tuple[float, ...] = self._validate_levels(
            fio2_levels if fio2_levels is not None
            else cfg.get("fio2_levels", list(FIO2_LEVELS)),
            name="fio2_levels", lo=0.21, hi=1.0)
        self.n_heater_levels: int = len(self.heater_levels)
        self.n_o2_levels: int = len(self.fio2_levels)
        self.n_actions: int = self.n_heater_levels * self.n_o2_levels

        # PID state (initialised by reset()).
        self._integral: float = 0.0
        self._prev_error: Optional[float] = None

        # PID internals exposed to the outside world (SHAP / logging).
        self.last_info: Dict[str, Any] = {}

        self.reset()

    # ------------------------------------------------------------------ #
    #  Grid helpers (Fix B)                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_levels(levels: Any, name: str, lo: float,
                         hi: float) -> tuple[float, ...]:
        """Validate an actuator level list; return it as a float tuple.

        Raises ``ValueError`` (never ``IndexError``) when the grid is not a
        sequence of >= 2 ascending values inside ``[lo, hi]``.
        """
        if not isinstance(levels, (list, tuple, np.ndarray)) or len(levels) < 2:
            raise ValueError(
                f"{name} must be a sequence of >= 2 values, got {levels!r}")
        vals = tuple(float(x) for x in levels)
        if list(vals) != sorted(vals) or len(set(vals)) != len(vals):
            raise ValueError(f"{name} must be strictly ascending, got {vals}")
        if not all(lo <= v <= hi for v in vals):
            raise ValueError(f"{name} must lie in [{lo}, {hi}], got {vals}")
        return vals

    @classmethod
    def from_env(cls, env: Any, **kwargs: Any) -> "RuleBasedBaselineController":
        """Build a controller whose action grid and ``dt`` MATCH ``env``.

        This is the recommended constructor whenever the environment was
        built with a custom ``config_path`` (e.g. a 3x2 grid): the baseline
        then encodes actions with the env's real ``n_o2_levels`` and can
        never emit an index outside ``env.action_space``.
        """
        kwargs.setdefault("heater_levels", list(env.heater_levels))
        kwargs.setdefault("fio2_levels", list(env.fio2_levels))
        kwargs.setdefault("dt", float(env.dt))
        ctrl = cls(**kwargs)
        n_env = int(env.action_space.n)
        if ctrl.n_actions != n_env:  # pragma: no cover - defensive
            raise ValueError(
                f"baseline grid ({ctrl.n_heater_levels}x{ctrl.n_o2_levels}="
                f"{ctrl.n_actions}) does not match env.action_space.n={n_env}")
        return ctrl

    def encode_action(self, heater_idx: int, o2_idx: int) -> int:
        """Combine channel indices -> flat action (same rule as the env)."""
        if not (0 <= heater_idx < self.n_heater_levels):
            raise ValueError(f"heater_idx {heater_idx} outside "
                             f"[0, {self.n_heater_levels})")
        if not (0 <= o2_idx < self.n_o2_levels):
            raise ValueError(f"o2_idx {o2_idx} outside [0, {self.n_o2_levels})")
        return int(heater_idx * self.n_o2_levels + o2_idx)

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear all PID state.

        Resets the integrator to zero, forgets the previous error (so the
        next call's derivative term is 0), and re-initialises the
        ``last_info`` dict so that no stale PID internals leak across
        episodes.
        """
        self._integral = 0.0
        self._prev_error = None   # signals "first call after reset"
        self.last_info = {
            "error": 0.0,
            "integral": 0.0,
            "derivative": 0.0,
            "output": 0.0,
            "heater_idx": 0,
            "o2_idx": 0,
            "heater_level": float(self.heater_levels[0]),
            "fio2_command": float(self.fio2_levels[0]),
            "target_Tc": self.target_Tc,
            "target_SpO2": self.target_SpO2,
        }

    # ------------------------------------------------------------------ #
    #  Per-channel controllers                                            #
    # ------------------------------------------------------------------ #
    def _pid_heater(
        self, Tc: float,
    ) -> tuple[int, float, float, float, float]:
        """Compute the heater channel.

        Parameters
        ----------
        Tc : float
            Current core temperature (deg C).

        Returns
        -------
        heater_idx : int
            Index into ``HEATER_LEVELS`` (0..4).
        output : float
            Continuous PID output in ``[0, 1]`` (after clipping).
        error : float
            ``target_Tc - Tc`` (positive -> too cold -> need heat).
        integral : float
            Current (post-clamp) integrator value.
        derivative : float
            Current derivative term (0 on the first call after reset).
        """
        error = self.target_Tc - Tc   # +ve = too cold, -ve = too hot

        # Derivative term (suppressed on the very first call after reset
        # so that the controller does not produce a huge spurious kick
        # from an uninitialized prev_error of 0).
        if self._prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / self.dt

        # Integrate with anti-windup clamping (clamp magnitude).
        new_integral = self._integral + error * self.dt
        self._integral = float(
            np.clip(new_integral, -self.integral_clamp, self.integral_clamp)
        )

        # Parallel-form PID output, clipped to the valid actuator range.
        # A feed-forward bias (FF_BIAS) compensates for the steady-state
        # ambient heat loss so that AT the setpoint (error=0) the controller
        # still supplies enough heater power to maintain Tc.  Without it,
        # the PID output is 0 at setpoint and the incubator cools.
        output = (
            self.FF_BIAS
            + self.Kp * error
            + self.Ki * self._integral
            + self.Kd * derivative
        )
        output = float(np.clip(output, 0.0, 1.0))

        # Snap to nearest discrete heater level.
        heater_idx = int(np.argmin(
            np.abs(np.asarray(self.heater_levels, dtype=np.float64) - output)
        ))

        # Stash for next derivative computation.
        self._prev_error = error
        return heater_idx, output, error, self._integral, derivative

    def _rule_fio2(self, SpO2: float) -> int:
        """Compute the FiO2 channel via the bang-bang rule-based mapper.

        V5-03 fix: indices are now computed DYNAMICALLY by selecting the
        configured FiO2 level whose value is CLOSEST to the clinically
        intended target. This preserves the EXACT default-grid behavior
        (5-level grid: severe→0.40 idx2, mild→0.30 idx1, normal→0.21 idx0)
        while supporting any custom action grid (2, 3, 4, ... levels)
        without IndexError.

        Strategy (intent targets, preserved across grid sizes):
          * Severe hypoxia (SpO2 < 89): pick the FiO2 closest to 0.40
            (the documented recovery level for the default grid).
          * Mild hypoxia (89 <= SpO2 < 91): pick the FiO2 closest to 0.30
            (the documented mild-boost level).
          * Normal band or hyperoxia (SpO2 >= 91): pick the LOWEST FiO2
            (index 0 = room air if configured) for energy saving.

        Parameters
        ----------
        SpO2 : float
            Current arterial oxygen saturation (%).

        Returns
        -------
        o2_idx : int
            Index into ``self.fio2_levels`` (0 .. n_o2_levels-1), guaranteed
            in range for any configured grid size.
        """
        if SpO2 < self.SPO2_SEVERE_HYPOXIA:
            # Severe hypoxia: pick the FiO2 closest to 0.40.
            target = 0.40
        elif SpO2 < self.SPO2_MILD_HYPOXIA:
            # Mild hypoxia: pick the FiO2 closest to 0.30.
            target = 0.30
        else:
            # Normal band or hyperoxia: room air (lowest configured FiO2).
            return 0
        levels = np.asarray(self.fio2_levels, dtype=np.float64)
        return int(np.argmin(np.abs(levels - target)))

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def select_action(self, state: np.ndarray) -> int:
        """Map an observation to a discrete action in ``[0, 24]``.

        Parameters
        ----------
        state : np.ndarray
            The 11-dim Gymnasium observation vector
            ``[Tc, Tsk, Ta, C_O2, SpO2, Tamb, dTa_dt, dC_O2_dt, phase,
            disturbance, scenario_id]`` (dtype float32).  Only ``Tc``
            (index 0) and ``SpO2`` (index 4) are consumed by the rule-based
            controller; the remaining channels are ignored.  For robustness,
            shorter state vectors of the legacy form
            ``[Tc, T_inc, SpO2, FiO2]`` are also accepted (Tc taken from
            index 0, SpO2 from index 2).

        Returns
        -------
        action : int
            Discrete action ``heater_idx * n_o2_levels + o2_idx`` in
            ``[0, n_heater*n_o2 - 1]`` (default 5x5 grid → ``[0, 24]``).

        Side effects
        ------------
        Updates ``self.last_info`` with the PID internals (``error``,
        ``integral``, ``derivative``, ``output``) and the resolved
        actuator indices / commands.
        """
        # Coerce to a 1-D float64 array for safe indexing.
        state = np.asarray(state, dtype=np.float64).reshape(-1)

        # Extract controlled variables -- support both the 11-dim
        # Gymnasium observation and the legacy 4-dim ``[Tc, Tinc, SpO2, FiO2]``
        # form so the controller stays usable from older test harnesses.
        if state.size >= 5:
            Tc = float(state[0])       # core temperature
            SpO2 = float(state[4])    # arterial O2 saturation
        elif state.size >= 3:
            Tc = float(state[0])
            SpO2 = float(state[2])
        elif state.size >= 1:
            Tc = float(state[0])
            SpO2 = self.target_SpO2
        else:
            Tc = self.target_Tc
            SpO2 = self.target_SpO2

        # Per-channel decisions.
        heater_idx, output, error, integral, derivative = self._pid_heater(Tc)
        o2_idx = self._rule_fio2(SpO2)

        # Combine into the env's single discrete action (validated).
        action = self.encode_action(heater_idx, o2_idx)

        # Expose PID internals + resolved commands for logging / SHAP.
        self.last_info = {
            # PID internals (required by the spec).
            "error": float(error),
            "integral": float(integral),
            "derivative": float(derivative),
            "output": float(output),
            # Resolved actuator commands (useful for logs / plots).
            "heater_idx": int(heater_idx),
            "o2_idx": int(o2_idx),
            "heater_level": float(self.heater_levels[heater_idx]),
            "fio2_command": float(self.fio2_levels[o2_idx]),
            # Inputs & setpoints (for traceability).
            "Tc": float(Tc),
            "SpO2": float(SpO2),
            "target_Tc": float(self.target_Tc),
            "target_SpO2": float(self.target_SpO2),
            "action": int(action),
        }
        return int(action)

    # ------------------------------------------------------------------ #
    #  Misc                                                               #
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return (
            f"RuleBasedBaselineController("
            f"target_Tc={self.target_Tc}, target_SpO2={self.target_SpO2}, "
            f"dt={self.dt}, Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}, "
            f"integral_clamp={self.integral_clamp})"
        )


# ---------------------------------------------------------------------------
#  Episode runner (used by __main__ and importable by other scripts)
# ---------------------------------------------------------------------------
def run_scenario(
    scenario: int,
    seed: int = 42,
    env_cls: Any = None,
) -> Dict[str, Any]:
    """Run one full episode of the baseline on the given scenario.

    Parameters
    ----------
    scenario : int
        Scenario index in ``{1, 2, 3, 4, 5}``.
    seed : int, default 42
        RNG seed passed to both the env constructor and ``env.reset``.
    env_cls : callable, optional
        Environment class to instantiate.  Defaults to
        ``NeonatalIncubatorEnv`` (imported lazily so this module can be
        imported without the env being available, e.g. for unit tests of
        the controller itself).

    Returns
    -------
    dict
        Episode summary with keys: ``scenario``, ``scenario_name``,
        ``total_reward``, ``mean_Tc``, ``mean_SpO2``,
        ``safety_violations``, ``final_Tc``, ``final_SpO2``,
        ``episode_length``, ``terminated``, ``truncated``.
    """
    if env_cls is None:
        # Late import to avoid a hard dependency at module load time.
        from environment import NeonatalIncubatorEnv, SCENARIO_NAMES
        env_cls = NeonatalIncubatorEnv
    else:
        from environment import SCENARIO_NAMES  # type: ignore

    env = env_cls(scenario=scenario, seed=seed)
    obs, info = env.reset(seed=seed)
    controller = RuleBasedBaselineController.from_env(env)
    controller.reset()

    total_reward: float = 0.0
    tcs: list[float] = []
    spo2s: list[float] = []
    safety_violations: int = 0
    steps: int = 0
    terminated: bool = False
    truncated: bool = False

    # Gymnasium 5-tuple step loop: stop on EITHER termination flag.
    while not (terminated or truncated):
        action = int(controller.select_action(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        tcs.append(float(info.get("Tc", float(obs[0]))))
        spo2s.append(float(info.get("SpO2", float(obs[4]))))
        # Count a safety violation on any step where the env applied the
        # -100 hard-limit penalty (i.e. r_safety < 0).
        if float(info.get("r_safety", 0.0)) < 0.0:
            safety_violations += 1
        steps += 1

    final_Tc = float(info.get("Tc", float(obs[0])))
    final_SpO2 = float(info.get("SpO2", float(obs[4])))

    return {
        "scenario": int(scenario),
        "scenario_name": SCENARIO_NAMES.get(scenario, f"scenario_{scenario}"),
        "total_reward": float(total_reward),
        "mean_Tc": float(np.mean(tcs)) if tcs else 0.0,
        "mean_SpO2": float(np.mean(spo2s)) if spo2s else 0.0,
        "safety_violations": int(safety_violations),
        "final_Tc": final_Tc,
        "final_SpO2": final_SpO2,
        "episode_length": int(steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def print_summary_table(results: list[Dict[str, Any]]) -> None:
    """Pretty-print a per-scenario summary table to stdout.

    Columns: scenario, total_reward, mean_Tc, mean_SpO2,
    safety_violations, final_Tc, final_SpO2, episode_length.
    """
    header = (
        f"{'scn':>3}  {'scenario_name':<30} "
        f"{'total_reward':>12} {'mean_Tc':>7} {'mean_SpO2':>9} "
        f"{'safety_v':>8} {'final_Tc':>8} {'final_SpO2':>10} "
        f"{'length':>6}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['scenario']:>3d}  {r['scenario_name']:<30s} "
            f"{r['total_reward']:>12.2f} {r['mean_Tc']:>7.3f} "
            f"{r['mean_SpO2']:>9.3f} {r['safety_violations']:>8d} "
            f"{r['final_Tc']:>8.3f} {r['final_SpO2']:>10.3f} "
            f"{r['episode_length']:>6d}"
        )
    print(sep)


# ---------------------------------------------------------------------------
#  Self-test smoke run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Make sibling src/ files importable regardless of the current working
    # directory (e.g. when run as ``python src/baseline.py`` from project
    # root, or as ``python baseline.py`` from inside ``src/``).
    _SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)

    from environment import NeonatalIncubatorEnv, SCENARIO_NAMES  # noqa: E402

    print("=" * 92)
    print("Neonatal Incubator -- Rule-based Baseline Controller smoke run")
    print("Gymnasium 5-tuple API: obs, r, terminated, truncated, info = env.step(a)")
    print("=" * 92)

    results: list[Dict[str, Any]] = []
    for scn in sorted(SCENARIO_NAMES):
        # V3-A3-fix: use the SAME per-scenario seed as run_simulation.py
        # episode 0 (seed + scenario*1000 + 0) so baseline.py numbers match
        # run_simulation.py --episodes 5 --seed 42 episode-0 values exactly.
        ep0_seed = 42 + scn * 1000
        results.append(run_scenario(scn, seed=ep0_seed, env_cls=NeonatalIncubatorEnv))

    print_summary_table(results)

    # Quick sanity diagnostics.
    print("\nController configuration:")
    ctrl = RuleBasedBaselineController()
    print(f"  {ctrl!r}")
    print(f"  heater_levels = {ctrl.heater_levels}")
    print(f"  fio2_levels   = {ctrl.fio2_levels}")
    print(f"  action range  = [0, {ctrl.n_heater_levels * ctrl.n_o2_levels - 1}]")
    print("=" * 92)
    print("Smoke test complete.")
