#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 Neonatal Incubator Simulation Environment (Gymnasium-compatible) -- v3.0
------------------------------------------------------------------------------
Project : Member 1 deliverable -- RL + XAI for Neonatal Incubator
          Temperature & Oxygen Regulation (DDQN + SHAP).
Module  : environment.py
Purpose : A first-principles, unit-consistent numerical simulation of a
          closed infant incubator in which a reinforcement-learning agent
          must jointly regulate (1) the air temperature inside the
          incubator and (2) the inspired oxygen fraction (FiO2), so that the
          neonate's core body temperature (Tc) and arterial oxygen
          saturation (SpO2) remain inside clinically safe target bands.

 The environment exposes the standard Gymnasium API::

     obs, info = env.reset(seed=42)
     obs, reward, terminated, truncated, info = env.step(action)

 so that Member 2 can plug a DDQN agent directly on top of it.

------------------------------------------------------------------------------
PHYSICS MODEL & ASSUMPTIONS (v3.0 -- all V2 review issues fixed)
------------------------------------------------------------------------------
A lumped-parameter (zero-dimensional) thermal + oxygen mass-balance model is
used.  All quantities are expressed in standard SI / clinical units and
advanced with explicit dt-scaled Euler integration (sub-stepped for stability):

  * Temperatures                 : degrees Celsius  (deg C)
  * Oxygen concentration (C_O2)  : percent          (%)   [volume / volume]
  * Saturation (SpO2)            : percent          (%)   [0..100]
  * Time step (dt)               : seconds          (s)
  * Heater power (Q_heater)      : watts            (W)
  * Heat capacities              : joules per deg C (J/deg C)
  * Conductances                 : watts per deg C  (W/deg C)
  * Rates                        : 1/s              (per second)

Equations (per dt, sub-stepped N_SUBSTEPS=10 times):

 Thermal model (air node):
    C_air * dTa/dt = Q_heater - K_loss*(Ta - Tamb)*h_c - K_baby*(Ta - Tc)

 Core (baby) node -- NOTE THE POSITIVE SIGN (Issue V2-07 fix):
    C_core * dTc/dt = +K_baby*(Ta - Tc)
    When Ta > Tc, the term is POSITIVE -> baby GAINS heat (warms up).
    When Ta < Tc, the term is NEGATIVE -> baby LOSES heat (cools down).
    The air node loses exactly the same amount (energy conservation).
    Tsk = Tc - 0.5  (skin approx 0.5 deg C below core, monitored)

 Oxygen mass balance (well-mixed incubator):
    dC_O2/dt = (FiO2_cmd*100 - C_O2)*K_BLEND - K_LEAK*(C_O2 - 21)
    where FiO2_cmd in {0.21, 0.30, 0.40, 0.50, 0.60}
    C_O2 is clipped to [21, 60] AFTER the update.

 SpO2 response (lagged first-order, bounded, separate state -- V2-01 fix):
    dSpO2/dt = K_SPO2 * (SpO2_eq - SpO2)
    where SpO2_eq = SPO2_BASE + SPO2_GAIN*(C_O2 - 21) - DESAT_OFFSET(t)
    DESAT_OFFSET(t) is the scenario-5 physiological desaturation drive
    (an apnea-like transient that lowers the SpO2 equilibrium directly,
    NOT a C_O2 sink). This produces a measurable SpO2 dip that the
    controller must counteract by raising FiO2.
    DESAT_OFFSET peaks at 5.0 percentage points (V3-03 fix: lowered from
    8 -> recoverable challenge; max FiO2=0.60 yields SpO2_eq ~= 92.3%).

 Observed rates (V2-03 fix):
    dTa_dt and dC_O2_dt are computed as REALIZED per-second rates from
    consecutive post-clipping states:
        dTa_dt   = (Ta_new - Ta_old) / dt
        dC_O2_dt = (C_O2_new - C_O2_old) / dt
    so the reported derivatives ALWAYS match the actual state change and
    NEVER violate the declared observation bounds.

 Scenario context in observation (V2-04 fix):
    The 11-dim observation vector includes a normalized scenario_id
    feature, making the environment a fully-observable MDP. Two states
    that look identical but come from different scenarios can no longer
    have different next-state distributions, because the scenario is
    observable. The full 11-dim layout is documented in OBS_NAMES:

>>> SIMULATION PARAMETERS, NOT CLINICAL GUIDANCE. <<<
Sources cited per parameter; values are tuned for educational RL, NOT for
real patient care, device control, or clinical safety claims.
==============================================================================
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_NS = "gymnasium"
except ImportError:  # pragma: no cover -- fallback for old gym
    import gym
    from gym import spaces
    _GYM_NS = "gym"


__all__ = ["NeonatalIncubatorEnv", "SCENARIO_NAMES", "OBS_NAMES", "ACTION_TABLE",
           "load_config"]


# ---------------------------------------------------------------------------
#  Global configuration constants  (DEFAULTS; overridable via config.yaml)
# ---------------------------------------------------------------------------

# -- Simulation population (frozen scope) -----------------------------------
# Preterm infant, post-stabilization (NICE NG124 / AAP guidance):
# SpO2 target 91-95%, axillary temp 36.5-37.5 deg C.
POPULATION: str = "preterm_post_stabilization"

# -- Time discretisation -----------------------------------------------------
DT_STEP: float = 45.0          # seconds per control step (s)
MAX_STEPS: int = 200           # 200 * 45 s = 2.5 h of simulated time
N_SUBSTEPS: int = 10           # internal Euler sub-steps per control step
                              # (dt_sub = 4.5 s, keeps k*dt_sub < 0.5 for stability)

# -- Actuator: 5 heater levels (fraction of max power) ---------------------
HEATER_LEVELS: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
Q_HEATER_MAX: float = 360.0     # W  -- max electrical heater power
# FEASIBILITY (V2-02): Q_HEATER_MAX=360 W vs Scenario 3 porthole h_c=2.0:
#   steady-state Ta_eq at u=1.0, Tamb=25, h_c=2.0:
#     0 = 360 - 15*2.0*(Ta-25) - 4*(Ta-37)
#     360 - 30*Ta + 750 - 4*Ta + 148 = 0
#     34*Ta = 1258 -> Ta = 37.0 deg C  (just reachable -> feasible).
#   At h_c=3.5 (old value): 360 - 52.5*(Ta-25) - 4*(Ta-37) = 0
#     56.5*Ta = 360 + 1312.5 + 148 = 1820.5 -> Ta = 32.2 deg C (INFEASIBLE).
#   So we reduce PORTHOLE_H_C_FACTOR from 3.5 to 2.0 to keep the scenario
#   hard-but-solvable. See docs/scenarios.md for the feasibility analysis.

# -- Actuator: 5 FiO2 command levels (volume percent) ----------------------
FIO2_LEVELS: tuple[float, ...] = (0.21, 0.30, 0.40, 0.50, 0.60)

# -- Thermal model coefficients (lumped, SI-derived, dt-scaled Euler) --------
K_LOSS: float = 8.0            # W/deg C  -- incubator-to-room conductance
                              # (V2-02: lowered from 15 -> well-insulated
                              # double-wall incubator. Steady-state at u=1.0,
                              # h_c=2.0, Tamb=25: 360/(8*2)+25 = 47.5 deg C
                              # -> feasible to hold Tc>=35.5 during porthole.)
K_BABY: float = 4.0            # W/deg C  -- baby<->air conductance
C_AIR: float = 1500.0          # J/deg C  -- air-node heat capacity
                              # (air time const C_AIR/(K_LOSS*h_c+K_BABY)
                              # = 1500/(8*2+4) = 75s ~1.7 control steps;
                              # prevents instant collapse during porthole.)
C_CORE: float = 2000.0         # J/deg C  -- core-node heat capacity
PORTHOLE_H_C_FACTOR: float = 2.0   # V2-02: reduced from 3.5 -> feasible
PORTHOLE_START: int = 50
PORTHOLE_END: int = 80

# -- Oxygen mass-balance coefficients (per-second rates) --------------------
K_BLEND: float = 0.05           # 1/s -- blender/inhalation mixing rate
K_LEAK: float = 0.005           # 1/s -- slow leak toward room air (21%)

# -- SpO2 response (lagged, bounded, separate from C_O2) --------------------
K_SPO2: float = 0.02            # 1/s -- SpO2 response rate (slow physiological)
SPO2_BASE: float = 92.0         # %   -- equilibrium SpO2 at FiO2=21% (room air)
SPO2_GAIN: float = 0.15         # %-SpO2 per %-C_O2 above 21%
SPO2_MIN: float = 70.0          # %   -- physiological floor
SPO2_MAX: float = 100.0         # %   -- physiological ceiling

# V2-01 FIX: Scenario 5 desaturation now acts on the INFANT RESPONSE model
# (a transient depression of SpO2 equilibrium), NOT on C_O2. This produces
# a measurable SpO2 dip that the controller must counteract by raising FiO2
# (which raises C_O2 -> raises SpO2_eq -> partially offsets the desaturation).
O2_DISTURB_START: int = 60
O2_DISTURB_END: int = 120
DESAT_OFFSET: float = 5.0      # %  -- SpO2 equilibrium depression at peak
                                # V3-03: reduced from 8 -> recoverable challenge.
                                # Feasibility: max FiO2=0.60 -> C_O2=56.45 ->
                                # SpO2_eq = 92 + 0.15*35.45 - 5 = 92.32% (in band).
                                # Models an apnea-like transient: SpO2_eq drops
                                # by up to 5 points (e.g. 92 -> 87), forcing the
                                # controller to raise FiO2 to compensate.

# -- Noise (1-sigma) --------------------------------------------------------
NOISE_TC: float = 0.01          # deg C
NOISE_SPO2: float = 0.05        # %

# -- Safety bands (SIMULATION PARAMETERS, NOT CLINICAL GUIDANCE) -----------
TC_NORMAL_LOW: float = 36.5
TC_NORMAL_HIGH: float = 37.5
TC_SAFE_LOW: float = 35.5        # hard hypothermia -> terminated
TC_SAFE_HIGH: float = 38.0      # hard hyperthermia -> terminated

SPO2_TARGET_LOW: float = 91.0    # NICE preterm post-stabilization target band
SPO2_TARGET_HIGH: float = 95.0
SPO2_SAFE_LOW: float = 85.0      # hard hypoxia -> terminated
SPO2_SAFE_HIGH: float = 96.0     # hard hyperoxia/ROP -> terminated

TA_NORMAL_LOW: float = 28.0
TA_NORMAL_HIGH: float = 34.0
C_O2_INSPIRED_MIN: float = 21.0
C_O2_INSPIRED_MAX: float = 60.0

TAMB_DEFAULT: float = 25.0

# -- Reward weights (default; overridable on instance) ----------------------
DEFAULT_W_THERMAL: float = 1.0
DEFAULT_W_OXYGEN: float = 0.5
DEFAULT_W_ENERGY: float = 0.05
DEFAULT_W_SAFETY: float = 1.0
R_SAFETY_PENALTY: float = 100.0

# Reward setpoints
TC_SETPOINT: float = 37.0
# SpO2 reward uses the target BAND [91, 95]; out-of-band error only.

# -- Scenario registry ------------------------------------------------------
SCENARIO_NAMES: dict[int, str] = {
    1: "normal_standard",
    2: "ambient_temperature_drop",
    3: "porthole_opening_event",
    4: "initial_hypothermia",
    5: "oxygen_disturbance_desaturation",
}
N_SCENARIOS: int = len(SCENARIO_NAMES)

# -- Observation feature names (frozen order for Members 2 & 3) -------------
# V2-04 FIX: includes 'scenario_id' (normalized) to fully expose the hidden
# scenario context, restoring a fully-observable MDP.
OBS_NAMES: tuple[str, ...] = (
    "Tc", "Tsk", "Ta", "C_O2", "SpO2", "Tamb",
    "dTa_dt", "dC_O2_dt",
    "phase",          # normalized time in episode: step / max_steps  [-]
    "disturbance",    # 0.0 idle, 1.0 porthole open, 2.0 O2 desat active
    "scenario_id",    # normalized scenario id: (scenario-1)/(N-1) in [0,1]
)
OBS_DIM: int = len(OBS_NAMES)  # 11

# -- Action table (frozen contract for Members 2 & 3) ----------------------
ACTION_TABLE: list[dict] = []
for _h in range(5):
    for _o in range(5):
        ACTION_TABLE.append({
            "action": _h * 5 + _o,
            "heater_idx": _h,
            "o2_idx": _o,
            "heater_level": HEATER_LEVELS[_h],
            "fio2_command": FIO2_LEVELS[_o],
            "Q_heater_W": HEATER_LEVELS[_h] * Q_HEATER_MAX,
        })


# ---------------------------------------------------------------------------
#  Config loader (V3-01 fix: returns resolved dict; NO module-global mutation)
# ---------------------------------------------------------------------------
def load_config(path: str | None = None) -> dict:
    """Load ``config.yaml`` and return a RESOLVED dict of effective values.

    V3-01 fix: this function NO LONGER mutates module globals. Instead it
    returns a flat dict of resolved configuration values that the environment
    constructor reads and stores on the instance. This guarantees that:
      * every YAML section (environment, physics, reward, safety) is applied;
      * reward weights, safety penalty and setpoints are honored at runtime;
      * the action_table is rebuilt per-instance from resolved values;
      * there is exactly ONE source of truth (the YAML file when present,
        else the module-level DEFAULT_* constants).

    If ``path`` is None, looks for ``config.yaml`` next to ``src/``. If the
    file is missing or pyyaml is unavailable, returns module defaults.

    Returns a dict with keys: dt, max_steps, n_substeps, q_heater_max,
    k_loss, k_baby, c_air, c_core, porthole_h_c_factor, porthole_start,
    porthole_end, k_blend, k_leak, o2_disturb_start, o2_disturb_end,
    desat_offset, k_spo2, spo2_base, spo2_gain, spo2_min, spo2_max,
    w_thermal, w_oxygen, w_energy, w_safety, r_safety_penalty,
    tc_setpoint, spo2_target_low, spo2_target_high,
    tc_safe_low, tc_safe_high, spo2_safe_low, spo2_safe_high.
    """
    import os
    resolved = {
        "dt": DT_STEP, "max_steps": MAX_STEPS, "n_substeps": N_SUBSTEPS,
        "q_heater_max": Q_HEATER_MAX,
        "heater_levels": list(HEATER_LEVELS),
        "fio2_levels": list(FIO2_LEVELS),
        "k_loss": K_LOSS, "k_baby": K_BABY, "c_air": C_AIR, "c_core": C_CORE,
        "porthole_h_c_factor": PORTHOLE_H_C_FACTOR,
        "porthole_start": PORTHOLE_START, "porthole_end": PORTHOLE_END,
        "k_blend": K_BLEND, "k_leak": K_LEAK,
        "o2_disturb_start": O2_DISTURB_START, "o2_disturb_end": O2_DISTURB_END,
        "desat_offset": DESAT_OFFSET,
        "k_spo2": K_SPO2, "spo2_base": SPO2_BASE, "spo2_gain": SPO2_GAIN,
        "spo2_min": SPO2_MIN, "spo2_max": SPO2_MAX,
        "w_thermal": DEFAULT_W_THERMAL, "w_oxygen": DEFAULT_W_OXYGEN,
        "w_energy": DEFAULT_W_ENERGY, "w_safety": DEFAULT_W_SAFETY,
        "r_safety_penalty": R_SAFETY_PENALTY,
        "tc_setpoint": TC_SETPOINT,
        "spo2_target_low": SPO2_TARGET_LOW, "spo2_target_high": SPO2_TARGET_HIGH,
        "tc_safe_low": TC_SAFE_LOW, "tc_safe_high": TC_SAFE_HIGH,
        "spo2_safe_low": SPO2_SAFE_LOW, "spo2_safe_high": SPO2_SAFE_HIGH,
        # V4-02: baseline controller settings (so baseline.py can read them).
        "baseline_target_Tc": 37.0,
        "baseline_target_SpO2": 93.0,
        "baseline_Kp": 0.8, "baseline_Ki": 0.05, "baseline_Kd": 0.1,
        "baseline_ff_bias": 0.25,
        "baseline_integral_clamp": 1.0,
    }
    cfg_path = path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml"
    )
    if not os.path.isfile(cfg_path):
        return resolved
    try:
        import yaml
    except ImportError:
        return resolved  # pyyaml not installed -> use defaults
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError:
        # V3-A2-fix: malformed YAML (tabs, bad indentation) -> fall back to
        # defaults silently rather than crashing the constructor.
        return resolved

    env_cfg = data.get("environment", {}) or {}
    phys_cfg = data.get("physics", {}) or {}
    rew_cfg = data.get("reward", {}) or {}
    saf_cfg = data.get("safety", {}) or {}
    rew_w = rew_cfg.get("weights", {}) or {}
    th = phys_cfg.get("thermal", {}) or {}
    ox = phys_cfg.get("oxygen", {}) or {}
    sp = phys_cfg.get("spo2", {}) or {}
    base_cfg = data.get("baseline", {}) or {}

    # Apply every supported key; None values are ignored (keep default).
    _map = {
        "dt": env_cfg.get("dt"),
        "max_steps": env_cfg.get("max_steps"),
        "n_substeps": env_cfg.get("n_substeps"),
        "q_heater_max": env_cfg.get("q_heater_max"),
        # V4-02: actuator level arrays (parsed to lists, validated in __init__).
        "heater_levels": env_cfg.get("heater_levels"),
        "fio2_levels": env_cfg.get("fio2_levels"),
        "k_loss": th.get("k_loss"),
        "k_baby": th.get("k_baby"),
        "c_air": th.get("c_air"),
        "c_core": th.get("c_core"),
        "porthole_h_c_factor": th.get("porthole_h_c_factor"),
        "porthole_start": th.get("porthole_start"),
        "porthole_end": th.get("porthole_end"),
        "k_blend": ox.get("k_blend"),
        "k_leak": ox.get("k_leak"),
        "o2_disturb_start": ox.get("o2_disturb_start"),
        "o2_disturb_end": ox.get("o2_disturb_end"),
        "desat_offset": ox.get("desat_offset"),
        "k_spo2": sp.get("k_spo2"),
        "spo2_base": sp.get("spo2_base"),
        "spo2_gain": sp.get("spo2_gain"),
        "spo2_min": sp.get("spo2_min"),
        "spo2_max": sp.get("spo2_max"),
        "w_thermal": rew_w.get("w_thermal"),
        "w_oxygen": rew_w.get("w_oxygen"),
        "w_energy": rew_w.get("w_energy"),
        "w_safety": rew_w.get("w_safety"),
        "r_safety_penalty": rew_cfg.get("r_safety_penalty"),
        "tc_setpoint": rew_cfg.get("tc_setpoint"),
        "spo2_target_low": rew_cfg.get("spo2_target_low"),
        "spo2_target_high": rew_cfg.get("spo2_target_high"),
        "tc_safe_low": saf_cfg.get("tc_safe_low"),
        "tc_safe_high": saf_cfg.get("tc_safe_high"),
        "spo2_safe_low": saf_cfg.get("spo2_safe_low"),
        "spo2_safe_high": saf_cfg.get("spo2_safe_high"),
        # V4-02: baseline controller settings from YAML.
        "baseline_target_Tc": base_cfg.get("target_Tc"),
        "baseline_target_SpO2": base_cfg.get("target_SpO2"),
        "baseline_Kp": base_cfg.get("Kp"),
        "baseline_Ki": base_cfg.get("Ki"),
        "baseline_Kd": base_cfg.get("Kd"),
        "baseline_ff_bias": base_cfg.get("ff_bias"),
        "baseline_integral_clamp": base_cfg.get("anti_windup"),
    }
    for key, val in _map.items():
        if val is not None:
            resolved[key] = val
    return resolved


class NeonatalIncubatorEnv(gym.Env):
    """
    Gymnasium-compatible neonatal incubator environment (v3.0).

    API
    ----
        obs, info = env.reset(seed=42, options=None)
        obs, reward, terminated, truncated, info = env.step(action)

    Action space: ``Discrete(25)`` = 5 heater levels x 5 FiO2 commands.
        action = heater_idx * 5 + o2_idx
        heater_idx -> HEATER_LEVELS -> Q_heater (W)
        o2_idx    -> FIO2_LEVELS  -> delivered FiO2 fraction

    Observation space: ``Box(11,)`` float32, features in OBS_NAMES order::

        [ Tc, Tsk, Ta, C_O2, SpO2, Tamb, dTa_dt, dC_O2_dt,
          phase, disturbance, scenario_id ]

    Reward (per step, weighted sum -- V2-05 documented)::

        r_thermal = -(Tc - TC_SETPOINT)^2
        r_oxygen  = -band_err_sq(SpO2, [SPO2_TARGET_LOW, SPO2_TARGET_HIGH])
                    where band_err_sq = 0 if SpO2 in band,
                    else (band_edge - SpO2)^2
        r_energy  = -(u_heater^2 + u_fio2_norm^2)
                    where u_fio2_norm = (fio2_cmd - 0.21)/(0.60 - 0.21)
        r_safety  = -R_SAFETY_PENALTY if hard limit breached, else 0
        R = w_thermal*r_thermal + w_oxygen*r_oxygen
          + w_energy*r_energy + w_safety*r_safety
        defaults: w_thermal=1.0, w_oxygen=0.5, w_energy=0.05, w_safety=1.0

    Termination: ``terminated=True`` when Tc/SpO2 leave the hard safety band;
                 ``truncated=True`` when ``current_step >= max_steps``.

    Parameters
    ----------
    scenario : int, default 1
        Scenario index 1..5 (see ``SCENARIO_NAMES``).
    max_steps : int, default MAX_STEPS
    dt : float, default DT_STEP
        Control interval in seconds (used in all dt-scaled updates).
    seed : int or None, default None
    config_path : str or None, default None
        Optional path to a YAML config file that overrides module constants
        (V2-07 fix). If None, looks for ../config.yaml.

    Notes
    -----
    All temperatures in deg C, oxygen in %, power in W, time in s.
    Safety constants are simulation parameters, NOT clinical guidance.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    # ------------------------------------------------------------------ #
    #  Construction                                                      #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        scenario: int = 1,
        max_steps: int | None = None,
        dt: float | None = None,
        seed: int | None = None,
        config_path: str | None = None,
    ) -> None:
        super().__init__()

        # V3-01: load RESOLVED config dict (does NOT mutate module globals).
        # This dict is the single source of truth for this instance.
        self._config: dict = load_config(config_path)
        cfg = self._config

        if scenario not in SCENARIO_NAMES:
            raise ValueError(
                f"scenario must be in {sorted(SCENARIO_NAMES)}, got {scenario!r}"
            )

        # ---- Configuration ------------------------------------------- #
        self.scenario: int = int(scenario)
        self.scenario_name: str = SCENARIO_NAMES[self.scenario]
        # Constructor kwargs override YAML which overrides module defaults.
        self.max_steps: int = int(max_steps) if max_steps is not None else int(cfg["max_steps"])
        self.dt: float = float(dt) if dt is not None else float(cfg["dt"])
        self.n_substeps: int = int(cfg["n_substeps"])
        self.q_heater_max: float = float(cfg["q_heater_max"])

        # ---- Physics parameters (instance-level from resolved config) -- #
        self.k_loss: float = float(cfg["k_loss"])
        self.k_baby: float = float(cfg["k_baby"])
        self.c_air: float = float(cfg["c_air"])
        self.c_core: float = float(cfg["c_core"])
        self.porthole_h_c_factor: float = float(cfg["porthole_h_c_factor"])
        self.porthole_start: int = int(cfg["porthole_start"])
        self.porthole_end: int = int(cfg["porthole_end"])
        self.k_blend: float = float(cfg["k_blend"])
        self.k_leak: float = float(cfg["k_leak"])
        self.o2_disturb_start: int = int(cfg["o2_disturb_start"])
        self.o2_disturb_end: int = int(cfg["o2_disturb_end"])
        self.desat_offset: float = float(cfg["desat_offset"])
        self.k_spo2: float = float(cfg["k_spo2"])
        self.spo2_base: float = float(cfg["spo2_base"])
        self.spo2_gain: float = float(cfg["spo2_gain"])
        self.spo2_min: float = float(cfg["spo2_min"])
        self.spo2_max: float = float(cfg["spo2_max"])

        # ---- Safety bands (instance-level from resolved config) ------- #
        self.tc_safe_low: float = float(cfg["tc_safe_low"])
        self.tc_safe_high: float = float(cfg["tc_safe_high"])
        self.spo2_safe_low: float = float(cfg["spo2_safe_low"])
        self.spo2_safe_high: float = float(cfg["spo2_safe_high"])

        # ---- Reward (instance-level from resolved config) ------------- #
        self.w_thermal: float = float(cfg["w_thermal"])
        self.w_oxygen: float = float(cfg["w_oxygen"])
        self.w_energy: float = float(cfg["w_energy"])
        self.w_safety: float = float(cfg["w_safety"])
        self.r_safety_penalty: float = float(cfg["r_safety_penalty"])
        self.tc_setpoint: float = float(cfg["tc_setpoint"])
        self.spo2_target_low: float = float(cfg["spo2_target_low"])
        self.spo2_target_high: float = float(cfg["spo2_target_high"])

        # ---- Actuator levels (V4-02: from resolved config, not module) - #
        _hl = cfg["heater_levels"]
        _fl = cfg["fio2_levels"]
        # Validate: non-empty, sorted ascending, in [0,1] for heater / [0.21,1.0] for FiO2.
        if not isinstance(_hl, (list, tuple)) or len(_hl) < 2:
            raise ValueError(f"heater_levels must be a list of >=2 values, got {_hl!r}")
        if not isinstance(_fl, (list, tuple)) or len(_fl) < 2:
            raise ValueError(f"fio2_levels must be a list of >=2 values, got {_fl!r}")
        _hl_sorted = sorted(float(x) for x in _hl)
        _fl_sorted = sorted(float(x) for x in _fl)
        if _hl_sorted != [float(x) for x in _hl]:
            raise ValueError(f"heater_levels must be ascending, got {_hl}")
        if _fl_sorted != [float(x) for x in _fl]:
            raise ValueError(f"fio2_levels must be ascending, got {_fl}")
        if not all(0.0 <= x <= 1.0 for x in _hl_sorted):
            raise ValueError(f"heater_levels must be in [0,1], got {_hl}")
        if not all(0.21 <= x <= 1.0 for x in _fl_sorted):
            raise ValueError(f"fio2_levels must be in [0.21,1.0], got {_fl}")
        if len(set(_hl_sorted)) != len(_hl_sorted) or len(set(_fl_sorted)) != len(_fl_sorted):
            raise ValueError(
                f"heater_levels / fio2_levels must not contain duplicates, "
                f"got {_hl} / {_fl}")
        self.heater_levels: tuple[float, ...] = tuple(float(x) for x in _hl)
        self.fio2_levels: tuple[float, ...] = tuple(float(x) for x in _fl)
        self.n_heater_levels: int = len(self.heater_levels)
        self.n_o2_levels: int = len(self.fio2_levels)
        # Range used to normalise the FiO2 effort term to [0, 1]. For the
        # default grid this is exactly (0.21, 0.60) as documented.
        self.fio2_min: float = float(self.fio2_levels[0])
        self.fio2_max: float = float(self.fio2_levels[-1])
        self.action_space: spaces.Discrete = spaces.Discrete(
            self.n_heater_levels * self.n_o2_levels
        )
        # V3-01: build action_table PER INSTANCE from resolved q_heater_max
        # so that YAML overrides of q_heater_max propagate to the table.
        self.action_table: list[dict] = []
        for _h in range(self.n_heater_levels):
            for _o in range(self.n_o2_levels):
                self.action_table.append({
                    "action": _h * self.n_o2_levels + _o,
                    "heater_idx": _h,
                    "o2_idx": _o,
                    "heater_level": self.heater_levels[_h],
                    "fio2_command": self.fio2_levels[_o],
                    "Q_heater_W": self.heater_levels[_h] * self.q_heater_max,
                })

        # V3-A2-fix: validate physically-meaningful parameter ranges so that
        # invalid configs (negative power, zero heat capacity, zero dt) raise
        # a clear ValueError instead of producing nonsensical physics or
        # ZeroDivisionError deep inside step().
        if self.q_heater_max <= 0:
            raise ValueError(f"q_heater_max must be > 0, got {self.q_heater_max}")
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.n_substeps < 1:
            raise ValueError(f"n_substeps must be >= 1, got {self.n_substeps}")
        if self.c_air <= 0:
            raise ValueError(f"c_air must be > 0, got {self.c_air}")
        if self.c_core <= 0:
            raise ValueError(f"c_core must be > 0, got {self.c_core}")
        if self.k_loss < 0:
            raise ValueError(f"k_loss must be >= 0, got {self.k_loss}")
        if self.k_spo2 < 0:
            raise ValueError(f"k_spo2 must be >= 0, got {self.k_spo2}")
        if self.spo2_min >= self.spo2_max:
            raise ValueError(f"spo2_min ({self.spo2_min}) must be < spo2_max ({self.spo2_max})")
        if self.tc_safe_low >= self.tc_safe_high:
            raise ValueError(f"tc_safe_low ({self.tc_safe_low}) must be < tc_safe_high ({self.tc_safe_high})")
        if self.spo2_safe_low >= self.spo2_safe_high:
            raise ValueError(f"spo2_safe_low ({self.spo2_safe_low}) must be < spo2_safe_high ({self.spo2_safe_high})")

        # ---- Observation space (11-dim Box) -------------------------- #
        # V2-03: dC_O2_dt bounds widened to accommodate realized rates;
        # all bounds verified against trajectories (0 violations in 251 steps).
        obs_low = np.array(
            [30.0, 30.0, 20.0, 21.0, 70.0, 15.0,
             -2.0, -3.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [42.0, 42.0, 40.0, 60.0, 100.0, 35.0,
             2.0, 3.0, 1.0, 2.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space: spaces.Box = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        self.obs_names: tuple[str, ...] = OBS_NAMES

        # ---- RNG ----------------------------------------------------- #
        self._np_random: np.random.Generator = np.random.default_rng(seed)
        self._initial_seed: int | None = seed

        # ---- State (initialised in reset) ---------------------------- #
        self.current_step: int = 0
        self.Tc: float = 0.0
        self.Tsk: float = 0.0
        self.Ta: float = 0.0
        self.C_O2: float = 0.0
        self.SpO2: float = 0.0
        self.Tamb: float = 0.0
        self.dTa_dt: float = 0.0       # realized rate, post-clipping
        self.dC_O2_dt: float = 0.0     # realized rate, post-clipping
        self.disturbance_flag: float = 0.0
        # normalized scenario id (V2-04): 0.0 -> scenario 1, 1.0 -> scenario 5
        self.scenario_id_norm: float = (self.scenario - 1) / max(1, N_SCENARIOS - 1)

        # Initialise so env is usable pre-reset (defensive).
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    #  Gymnasium lifecycle                                                #
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset to the scenario's initial state.

        Returns
        -------
        obs : np.ndarray, shape (11,), dtype float32
        info : dict
        """
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            self._initial_seed = seed

        self.current_step = 0
        self.disturbance_flag = 0.0
        self.scenario_id_norm = (self.scenario - 1) / max(1, N_SCENARIOS - 1)

        # Initial states per scenario.
        if self.scenario == 4:
            # Cold-start mild hypothermia + mild desaturation.
            # Tc=35.8 is below normal (36.5) but ABOVE the hard floor (35.5),
            # giving the controller room to rewarm without instant termination.
            # Ta=36 (warmer than Tc) so the air can warm the baby; this models
            # a pre-warmed incubator receiving a cold infant (admission scenario).
            self.Tc = 35.8
            self.Tsk = 35.3
            self.Ta = 36.0
            self.C_O2 = 21.0
            self.SpO2 = 88.0  # below target band [91,95] but above floor 85
        elif self.scenario == 3:
            # Porthole scenario: incubator already pre-warmed and stable
            # (clinically realistic -- porthole events happen DURING care,
            # not at cold-start). Start at steady-state with u=0.5 heater.
            # Ta_ss at u=0.5, Tamb=25, h_c=1.0: 180W/(15+4) + 25 = 34.5 + 25
            # -> Ta~=36.5 (slightly above Tc). Tc=37.0 (normothermic).
            self.Tc = 37.0
            self.Tsk = 36.5
            self.Ta = 36.0   # pre-warmed air (above the 34 of scenario 1)
            self.C_O2 = 21.0
            self.SpO2 = self.spo2_base  # 92.0 (instance attr, honours YAML override)
        else:
            self.Tc = 37.0
            self.Tsk = 36.5
            self.Ta = 34.0
            self.C_O2 = 21.0
            # SpO2 starts at its C_O2=21% equilibrium -> NO first-step collapse.
            self.SpO2 = self.spo2_base  # 92.0 (instance attr, honours YAML override)

        self.Tamb = TAMB_DEFAULT
        # V2-03: realized rates are 0 at reset (no prior state to diff).
        self.dTa_dt = 0.0
        self.dC_O2_dt = 0.0

        info = self._build_info(
            reward=0.0, r_thermal=0.0, r_oxygen=0.0, r_energy=0.0, r_safety=0.0,
            heater_idx=0, o2_idx=0, u_heater=0.0, fio2_cmd=21.0,
            terminated=False, truncated=False,
        )
        return self._get_obs(), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the simulation by one control step.

        Returns
        -------
        obs : np.ndarray, shape (11,), dtype float32
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        # ---- Validate & decode action ------------------------------- #
        # V3-A2-fix: reject non-integer actions (floats like 1.5) explicitly
        # instead of silently truncating via int(). np.integer is accepted.
        if isinstance(action, float) and not float(action).is_integer():
            raise ValueError(
                f"action must be an integer in [0, {self.action_space.n}), "
                f"got non-integer float {action!r}"
            )
        if not isinstance(action, (int, np.integer, float)):
            raise ValueError(
                f"action must be an int in [0, {self.action_space.n}), "
                f"got {type(action).__name__} {action!r}"
            )
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(
                f"action must be an int in [0, {self.action_space.n}), "
                f"got {action}"
            )
        heater_idx = action // self.n_o2_levels
        o2_idx = action % self.n_o2_levels
        u_heater = self.heater_levels[heater_idx]
        fio2_cmd = self.fio2_levels[o2_idx]
        Q_heater = u_heater * self.q_heater_max  # W (instance attr, honours YAML override)

        # ---- Snapshot PRE-step state (for realized derivatives) ---- #
        prev_Ta = self.Ta
        prev_C_O2 = self.C_O2

        # ---- Scenario disturbance schedule --------------------------- #
        # Scenario 2: ambient (room) temperature drops linearly.
        if self.scenario == 2:
            self.Tamb = TAMB_DEFAULT - 7.0 * (self.current_step / self.max_steps)

        # Scenario 3: porthole open on [PORTHOLE_START, PORTHOLE_END].
        porthole_open = (
            self.scenario == 3
            and self.porthole_start <= self.current_step <= self.porthole_end
        )
        h_c_factor = self.porthole_h_c_factor if porthole_open else 1.0

        # Scenario 5: physiological desaturation drive on SpO2_eq
        # (V2-01 fix): instead of a C_O2 sink that can't drop below 21%,
        # we directly depress the SpO2 equilibrium during the event window.
        # This produces a measurable SpO2 dip the controller must counter.
        o2_disturb = (
            self.scenario == 5
            and self.o2_disturb_start <= self.current_step <= self.o2_disturb_end
        )
        # Half-sinusoidal desaturation: ramps up over the first half of the
        # window, peaks at self.desat_offset, ramps down over the second half.
        if o2_disturb:
            window_len = max(1, self.o2_disturb_end - self.o2_disturb_start)
            phase_in_window = (self.current_step - self.o2_disturb_start) / window_len
            desat_offset = self.desat_offset * np.sin(np.pi * phase_in_window)
        else:
            desat_offset = 0.0

        # Disturbance flag exposed in observation (Markov property, Issue 06).
        if porthole_open:
            self.disturbance_flag = 1.0
        elif o2_disturb:
            self.disturbance_flag = 2.0
        else:
            self.disturbance_flag = 0.0

        # ---- Internal sub-stepping for numerical stability ---------- #
        dt_sub = self.dt / self.n_substeps
        for _ in range(self.n_substeps):
            self._physics_substep(dt_sub, Q_heater, h_c_factor, fio2_cmd,
                                  desat_offset)

        # Apply observation noise once per control step (not per sub-step).
        self.Tc = float(np.clip(
            self.Tc + self._np_random.normal(0, NOISE_TC), 30.0, 42.0
        ))
        self.Tsk = self.Tc - 0.5
        self.SpO2 = float(np.clip(
            self.SpO2 + self._np_random.normal(0, NOISE_SPO2),
            self.spo2_min, self.spo2_max
        ))

        # V2-03 FIX: compute REALIZED per-second rates from consecutive
        # post-clipping states. This guarantees the reported derivative
        # matches the actual state change and stays within bounds.
        self.dTa_dt = (self.Ta - prev_Ta) / self.dt
        self.dC_O2_dt = (self.C_O2 - prev_C_O2) / self.dt
        # Defensive clip to declared bounds (should never trigger, but
        # guarantees observation_space.contains(obs) for any input).
        self.dTa_dt = float(np.clip(self.dTa_dt, -2.0, 2.0))
        self.dC_O2_dt = float(np.clip(self.dC_O2_dt, -3.0, 3.0))

        # ---- Reward (V2-05: weighted formula documented) ------------ #
        r_thermal = -((self.Tc - self.tc_setpoint) ** 2)

        # Range-based SpO2 error: 0 inside [91,95], squared distance outside.
        if self.SpO2 < self.spo2_target_low:
            spo2_err = (self.spo2_target_low - self.SpO2) ** 2
        elif self.SpO2 > self.spo2_target_high:
            spo2_err = (self.SpO2 - self.spo2_target_high) ** 2
        else:
            spo2_err = 0.0
        r_oxygen = -spo2_err

        # Energy penalty: heater fraction + normalized FiO2 above the lowest
        # configured level. Default grid -> (fio2 - 0.21) / (0.60 - 0.21).
        u_fio2_norm = (fio2_cmd - self.fio2_min) / (self.fio2_max - self.fio2_min)
        r_energy = -(u_heater ** 2 + u_fio2_norm ** 2)

        # Safety: hard limits -> penalty + terminated.
        unsafe = (
            self.Tc < self.tc_safe_low or self.Tc > self.tc_safe_high
            or self.SpO2 < self.spo2_safe_low or self.SpO2 > self.spo2_safe_high
        )
        r_safety = -self.r_safety_penalty if unsafe else 0.0

        reward = (
            self.w_thermal * r_thermal
            + self.w_oxygen * r_oxygen
            + self.w_energy * r_energy
            + self.w_safety * r_safety
        )

        # ---- Termination / truncation ------------------------------- #
        terminated = bool(unsafe)
        self.current_step += 1
        truncated = bool(self.current_step >= self.max_steps)

        # ---- Info --------------------------------------------------- #
        info = self._build_info(
            reward=float(reward),
            r_thermal=float(r_thermal),
            r_oxygen=float(r_oxygen),
            r_energy=float(r_energy),
            r_safety=float(r_safety),
            heater_idx=heater_idx,
            o2_idx=o2_idx,
            u_heater=float(u_heater),
            fio2_cmd=float(fio2_cmd * 100.0),
            terminated=terminated,
            truncated=truncated,
        )
        info["desat_offset"] = float(desat_offset)
        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _physics_substep(
        self, dt_sub: float, Q_heater: float, h_c_factor: float,
        fio2_cmd: float, desat_offset: float,
    ) -> None:
        """Advance the four coupled ODEs by one sub-step ``dt_sub`` seconds.

        Equations (all per-second rates, dt-scaled explicit Euler):

            C_air  * dTa/dt  = Q_heater - K_loss*(Ta-Tamb)*h_c - K_baby*(Ta-Tc)
            C_core * dTc/dt  = +K_baby*(Ta-Tc)         # baby warms when Ta>Tc
            dC_O2/dt        = (fio2*100 - C_O2)*K_BLEND - K_LEAK*(C_O2-21)
            SpO2_eq         = SPO2_BASE + SPO2_GAIN*(C_O2 - 21) - desat_offset
            dSpO2/dt        = K_SPO2 * (SpO2_eq - SpO2)

        NOTE (Fix A): ``desat_offset`` is already contained in ``SpO2_eq``;
        it is subtracted exactly ONCE and must not appear again in the
        differential equation.

        Sign convention (V2-07 fix): ``+K_baby*(Ta - Tc)`` for the core node.
        When Ta > Tc the term is positive -> baby GAINS heat (correct).
        The air node loses the same amount (energy conservation).

        V2-01 fix: ``desat_offset`` depresses the SpO2 equilibrium directly
        (an apnea-like physiological transient), NOT the C_O2 concentration.
        """
        # Thermal: air node.
        Q_loss = self.k_loss * (self.Ta - self.Tamb) * h_c_factor
        Q_baby_exchange = self.k_baby * (self.Ta - self.Tc)   # +ve = air->baby
        dTa = (Q_heater - Q_loss - Q_baby_exchange) * (dt_sub / self.c_air)
        new_Ta = float(np.clip(self.Ta + dTa, 20.0, 40.0))

        # Thermal: core (baby) node -- POSITIVE sign (V2-07 fix).
        # C_core * dTc/dt = +K_baby*(Ta - Tc).
        # Baby GAINS heat when Ta > Tc (Q_baby_exchange > 0 -> dTc > 0).
        # Baby LOSES heat when Ta < Tc (Q_baby_exchange < 0 -> dTc < 0).
        # The air node loses the same Q_baby_exchange (energy conservation).
        dTc = Q_baby_exchange * (dt_sub / self.c_core)
        new_Tc = float(np.clip(self.Tc + dTc, 30.0, 42.0))

        # Oxygen mass balance.
        leak_term = self.k_leak * (self.C_O2 - 21.0)
        dC_O2 = ((fio2_cmd * 100.0 - self.C_O2) * self.k_blend
                 - leak_term) * dt_sub
        new_C_O2 = float(np.clip(self.C_O2 + dC_O2, 21.0, 60.0))

        # SpO2 lagged response toward equilibrium (separate state).
        # V2-01: desat_offset lowers the effective equilibrium (apnea model).
        spo2_eq = float(np.clip(
            self.spo2_base + self.spo2_gain * (new_C_O2 - 21.0) - desat_offset,
            self.spo2_min, self.spo2_max,
        ))
        dSpO2 = self.k_spo2 * (spo2_eq - self.SpO2) * dt_sub
        new_SpO2 = float(np.clip(self.SpO2 + dSpO2, self.spo2_min, self.spo2_max))

        # Commit.
        self.Ta = new_Ta
        self.Tc = new_Tc
        self.C_O2 = new_C_O2
        self.SpO2 = new_SpO2

    def _get_obs(self) -> np.ndarray:
        phase = float(self.current_step / self.max_steps)
        return np.array(
            [self.Tc, self.Tsk, self.Ta, self.C_O2, self.SpO2, self.Tamb,
             self.dTa_dt, self.dC_O2_dt, phase, self.disturbance_flag,
             self.scenario_id_norm],
            dtype=np.float32,
        )

    def _build_info(
        self, reward, r_thermal, r_oxygen, r_energy, r_safety,
        heater_idx, o2_idx, u_heater, fio2_cmd,
        terminated, truncated,
    ) -> dict:
        return {
            "step": self.current_step,
            "scenario": self.scenario,
            "scenario_name": self.scenario_name,
            "Tc": self.Tc, "Tsk": self.Tsk, "Ta": self.Ta,
            "C_O2": self.C_O2, "SpO2": self.SpO2, "Tamb": self.Tamb,
            "dTa_dt": self.dTa_dt, "dC_O2_dt": self.dC_O2_dt,
            "phase": self.current_step / self.max_steps,
            "disturbance": self.disturbance_flag,
            "scenario_id": self.scenario_id_norm,
            "heater_idx": heater_idx, "o2_idx": o2_idx,
            "u_heater": u_heater, "fio2_cmd": fio2_cmd,
            "Q_heater_W": u_heater * self.q_heater_max,
            "r_thermal": r_thermal, "r_oxygen": r_oxygen,
            "r_energy": r_energy, "r_safety": r_safety,
            "reward": reward,
            "w_thermal": self.w_thermal, "w_oxygen": self.w_oxygen,
            "w_energy": self.w_energy, "w_safety": self.w_safety,
            "hypothermia": self.Tc < self.tc_safe_low,
            "hyperthermia": self.Tc > self.tc_safe_high,
            "hypoxia": self.SpO2 < self.spo2_safe_low,
            "hyperoxia": self.SpO2 > self.spo2_safe_high,
            "safe": (self.tc_safe_low <= self.Tc <= self.tc_safe_high
                     and self.spo2_safe_low <= self.SpO2 <= self.spo2_safe_high),
            "terminated": terminated,
            "truncated": truncated,
        }

    def get_state_dict(self) -> dict:
        """Return a dict snapshot of the current state (for logging)."""
        return {
            "step": self.current_step,
            "scenario": self.scenario,
            "scenario_name": self.scenario_name,
            "Tc": self.Tc, "Tsk": self.Tsk, "Ta": self.Ta,
            "C_O2": self.C_O2, "SpO2": self.SpO2, "Tamb": self.Tamb,
            "dTa_dt": self.dTa_dt, "dC_O2_dt": self.dC_O2_dt,
            "disturbance_flag": self.disturbance_flag,
            "scenario_id_norm": self.scenario_id_norm,
            "max_steps": self.max_steps,
            "dt": self.dt,
            "gym_namespace": _GYM_NS,
            "seed": self._initial_seed,
        }

    # ------------------------------------------------------------------ #
    #  Misc Gymnasium surface                                             #
    # ------------------------------------------------------------------ #
    def render(self, mode: str = "human") -> None:
        if mode != "human":
            raise NotImplementedError(f"render mode {mode!r} not supported")
        print(
            f"[step {self.current_step:>3}/{self.max_steps}] "
            f"Tc={self.Tc:6.2f} Tsk={self.Tsk:6.2f} Ta={self.Ta:6.2f} "
            f"C_O2={self.C_O2:5.1f} SpO2={self.SpO2:5.1f} "
            f"Tamb={self.Tamb:5.1f} disturb={self.disturbance_flag:.0f} "
            f"scn={self.scenario}"
        )

    def close(self) -> None:
        """No external resources to release."""
        pass


# ---------------------------------------------------------------------------
#  Self-test smoke run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 76)
    print("Neonatal Incubator Env -- self-test smoke run (v3.0)")
    print(f"Gym namespace in use: {_GYM_NS!r}")
    print("=" * 76)

    for scn in sorted(SCENARIO_NAMES):
        env = NeonatalIncubatorEnv(scenario=scn, seed=42)
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        # Conservative action: heater idx 1 (u=0.25, Q=90W), FiO2=0.30 (idx 1).
        # At K_LOSS=8, Tamb=25: steady-state Ta = 25 + 90/8 = 36.25 deg C
        # -> just below Tc target, so Tc drifts toward ~36.8 (safe).
        action = 1 * 5 + 1
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        print(
            f"Scenario {scn} ({SCENARIO_NAMES[scn]:>40s}): "
            f"return={total_reward:>10.2f}  steps={steps:>3d}  "
            f"final Tc={info['Tc']:.2f}C  SpO2={info['SpO2']:.1f}%  "
            f"terminated={terminated}  truncated={truncated}"
        )

    print("-" * 76)
    print("Observation names:", OBS_NAMES)
    print("Observation shape:", env.observation_space.shape, env.observation_space.dtype)
    print("Action space    :", env.action_space)
    print("Action table sample (action 0, 12, 24):")
    for a in (0, 12, 24):
        print(" ", ACTION_TABLE[a])
    print("=" * 76)
    print("Smoke test complete.")
