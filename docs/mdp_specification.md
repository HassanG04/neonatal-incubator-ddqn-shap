# MDP Specification — Neonatal Incubator (v3.1)

> Version 3.1 — all eight V2 review issues (V2-01 … V2-08) resolved,
> plus five V3 review issues (V3-01 … V3-05).
> This document is the **normative** specification of the Markov Decision
> Process (MDP) implemented in `src/environment.py` and configured by
> `config.yaml` (single source of truth, V2-07).
> All numbers below are **simulation parameters**, not clinical guidance.

---

## 1. Overview

The Neonatal Incubator environment models a closed radiant / convective
incubator in which a reinforcement-learning agent must jointly regulate two
controlled variables of a preterm, post-stabilisation neonate:

* **Core body temperature** `Tc` (target 37.0 °C, comfort band 36.5–37.5 °C),
  driven by the heater power `Q_heater ∈ {0, 90, 180, 270, 360} W`;
* **Arterial oxygen saturation** `SpO2` (target band 91–95 %, hard band 85–96 %),
  driven by the inspired oxygen fraction `FiO2 ∈ {0.21, 0.30, 0.40, 0.50, 0.60}`.

The dynamics are first-principles lumped-parameter ODEs (thermal + oxygen mass
balance + lagged SpO2 response) integrated by dt-scaled explicit Euler with
`N_SUBSTEPS = 10` sub-steps per control step (`dt = 45 s`, `dt_sub = 4.5 s`),
which keeps `k·dt_sub < 0.5` for every rate in the model and guarantees
numerical stability. The environment exposes the standard **Gymnasium API**
(`obs, info = env.reset(seed=42)`; `obs, r, terminated, truncated, info =
env.step(action)`), and is therefore directly pluggable into Member 2's DDQN
trainer and Member 3's SHAP explainer.

---

## 2. State Space

The observation is an 11-dimensional `Box` of `float32` (V2-04 fix: the
`scenario_id` feature has been added so that two otherwise-identical states
that originate from different scenarios are no longer aliased; the MDP is
fully observable).

| # | Feature | Unit | Range | Description |
|---|---------|------|-------|-------------|
| 0 | `Tc` | °C | [30, 42] | Core body temperature (baby node) |
| 1 | `Tsk` | °C | [30, 42] | Skin temperature (`= Tc − 0.5`) |
| 2 | `Ta` | °C | [20, 40] | Incubator air temperature (air node) |
| 3 | `C_O2` | % | [21, 60] | Inspired O2 concentration (well-mixed) |
| 4 | `SpO2` | % | [70, 100] | Arterial O2 saturation (separate lagged state, V2-01) |
| 5 | `Tamb` | °C | [15, 35] | Ambient (room) temperature |
| 6 | `dTa_dt` | °C/s | [−2, +2] | Realized rate of `Ta` (post-clipping, V2-03) |
| 7 | `dC_O2_dt` | %/s | [−3, +3] | Realized rate of `C_O2` (post-clipping, V2-03) |
| 8 | `phase` | − | [0, 1] | Normalised episode time `step / max_steps` |
| 9 | `disturbance` | − | {0, 1, 2} | 0 idle, 1 porthole open, 2 O2 desaturation active |
| 10 | `scenario_id` | − | [0, 1] | Normalised scenario id `(scenario−1)/4` (V2-04) |

The feature order is **frozen** by `OBS_NAMES` in `environment.py` and
mirrored by `environment.obs_names` in `config.yaml`. Member 2 must use this
exact 11-feature vector as input to the Q-network, and Member 3 must use the
same 11 feature names when attributing SHAP values.

---

## 3. Action Space

The action space is `Discrete(25) = 5 heater levels × 5 FiO2 commands`
(the V2 review confirmed absolute FiO2 fractions, not normalised values).

Action decoding: `action = heater_idx × 5 + o2_idx`.

| action | heater_idx | o2_idx | heater_level | fio2_command | Q_heater (W) |
|--------|------------|--------|--------------|--------------|-------------|
| 0  | 0 | 0 | 0.00 | 0.21 |   0 |
| 1  | 0 | 1 | 0.00 | 0.30 |   0 |
| 2  | 0 | 2 | 0.00 | 0.40 |   0 |
| 3  | 0 | 3 | 0.00 | 0.50 |   0 |
| 4  | 0 | 4 | 0.00 | 0.60 |   0 |
| 5  | 1 | 0 | 0.25 | 0.21 |  90 |
| 6  | 1 | 1 | 0.25 | 0.30 |  90 |
| 7  | 1 | 2 | 0.25 | 0.40 |  90 |
| 8  | 1 | 3 | 0.25 | 0.50 |  90 |
| 9  | 1 | 4 | 0.25 | 0.60 |  90 |
| 10 | 2 | 0 | 0.50 | 0.21 | 180 |
| 11 | 2 | 1 | 0.50 | 0.30 | 180 |
| 12 | 2 | 2 | 0.50 | 0.40 | 180 |
| 13 | 2 | 3 | 0.50 | 0.50 | 180 |
| 14 | 2 | 4 | 0.50 | 0.60 | 180 |
| 15 | 3 | 0 | 0.75 | 0.21 | 270 |
| 16 | 3 | 1 | 0.75 | 0.30 | 270 |
| 17 | 3 | 2 | 0.75 | 0.40 | 270 |
| 18 | 3 | 3 | 0.75 | 0.50 | 270 |
| 19 | 3 | 4 | 0.75 | 0.60 | 270 |
| 20 | 4 | 0 | 1.00 | 0.21 | 360 |
| 21 | 4 | 1 | 1.00 | 0.30 | 360 |
| 22 | 4 | 2 | 1.00 | 0.40 | 360 |
| 23 | 4 | 3 | 1.00 | 0.50 | 360 |
| 24 | 4 | 4 | 1.00 | 0.60 | 360 |

`fio2_command` is the **absolute** volume fraction delivered to the blender
(`0.21 = 21 %`, room air; `0.60 = 60 %`). The oxygen mass-balance equation
uses `FiO2·100` to convert to `%`.

---

## 4. Reward Function (V2-05 — exact weighted formula)

The per-step reward is the weighted sum of four shaping terms. All constants
live in `config.yaml → reward` (single source of truth, V2-07).

**Thermal** (target 37.0 °C):
```
r_thermal = -(Tc - 37.0)²
```

**Oxygen** (target band [91, 95] %; in-band ⇒ 0; out-of-band ⇒ squared edge error):
```
                    ⎧ 0                         if SpO2 ∈ [91, 95]
band_err_sq(SpO2) = ⎨ (91 - SpO2)²              if SpO2 < 91
                    ⎩ (SpO2 - 95)²              if SpO2 > 95
r_oxygen = -band_err_sq(SpO2, [91, 95])
```

**Energy** (penalise actuator usage):
```
u_fio2_norm = (fio2_cmd - 0.21) / (0.60 - 0.21)        # ∈ [0, 1]
r_energy    = -(u_heater² + u_fio2_norm²)
```

**Safety** (hard breach only):
```
r_safety = -100  if Tc ∉ [35.5, 38.0] OR SpO2 ∉ [85, 96]
         =    0  otherwise
```

**Aggregation** (default weights from `config.yaml`):
```
R = 1.0·r_thermal + 0.5·r_oxygen + 0.05·r_energy + 1.0·r_safety
```

### Worked example (V2-05)

At `Tc = 37.0`, `SpO2 = 93.0`, `u_heater = 0.5`, `fio2_cmd = 0.30`, safe:

```
r_thermal = -(37.0 - 37.0)²                  =   0.000
r_oxygen  = -0   (93 ∈ [91, 95])             =   0.000
u_fio2_norm = (0.30 - 0.21)/0.39             =   0.231
r_energy  = -(0.5² + 0.231²)                 =  -0.303
r_safety  = 0  (safe)                        =   0.000
R = 1.0·0 + 0.5·0 + 0.05·(-0.303) + 1.0·0    =  -0.0152
```

---

## 5. Transition Dynamics

The four coupled ODEs are advanced with explicit dt-scaled Euler inside
`NeonatalIncubatorEnv._physics_substep`, called `N_SUBSTEPS = 10` times per
control step. All temperatures are in °C, oxygen in %, power in W, time in s.

### 5.1 Thermal model (air + core nodes)

```
C_air  · dTa/dt = Q_heater - K_loss · (Ta - Tamb) · h_c - K_baby · (Ta - Tc)
C_core · dTc/dt = +K_baby · (Ta - Tc)         # POSITIVE sign (V2-07 fix)
Tsk ≈ Tc - 0.5
```

Sign convention: when `Ta > Tc`, the exchange term `+K_baby·(Ta−Tc)` is
**positive**, i.e. the baby **gains** heat (warms up) and the air node loses
exactly the same quantity (energy conservation). The V2 review caught a
sign-flip bug in an earlier draft; the present `+` sign is now correct,
documented in the module docstring, and exercised by `tests/test_environment.py`.

| Symbol | Value | Unit | Description |
|--------|------:|------|-------------|
| `K_LOSS` | 8.0 | W/°C | Incubator-to-room conductance (V2-02 lowered from 15) |
| `K_BABY` | 4.0 | W/°C | Baby↔air conductance |
| `C_AIR` | 1500.0 | J/°C | Air-node heat capacity (V2-02 raised from 200) |
| `C_CORE` | 2000.0 | J/°C | Core-node heat capacity |
| `PORTHOLE_H_C_FACTOR` | 2.0 | − | Porthole multiplier on `K_LOSS` (V2-02 lowered from 3.5) |

### 5.2 Oxygen mass balance

```
dC_O2/dt = (FiO2·100 - C_O2) · K_BLEND - K_LEAK · (C_O2 - 21)
C_O2 ← clip(C_O2, 21, 60)    # post-update clipping
```

`K_BLEND = 0.05 1/s` is the blender/inhalation mixing rate;
`K_LEAK = 0.005 1/s` is the slow leak toward room air (21 %).

### 5.3 SpO2 lagged response (V2-01 fix — separate state)

```
SpO2_eq = SPO2_BASE + SPO2_GAIN · (C_O2 - 21) - desat_offset(t)
dSpO2/dt = K_SPO2 · (SpO2_eq - SpO2)
SpO2 ← clip(SpO2, SPO2_MIN, SPO2_MAX)
```

`SpO2` is a **separate lagged state** with its own ODE — it does **not**
collapse on the first step (which was a V2-01 bug in earlier drafts that used
`SpO2 := SpO2_eq` directly). `K_SPO2 = 0.02 1/s`,
`SPO2_BASE = 92 %`, `SPO2_GAIN = 0.15`.

### 5.4 Realized rates (V2-03 fix)

The two observed rates in the observation are **realized** per-second rates
computed from consecutive post-clipping states:

```
dTa_dt   = (Ta_new - Ta_old) / dt
dC_O2_dt = (C_O2_new - C_O2_old) / dt
```

These derivatives are then defensively clipped to the declared
`observation_space` bounds (`[-2, +2]` and `[-3, +3]` respectively). This
guarantees `observation_space.contains(obs)` for every step, eliminating the
out-of-bounds derivatives reported in the V2 review.

### 5.5 Desaturation disturbance (V2-01 fix)

For scenario 5 only, on steps `[60, 120]`:

```
desat_offset(t) = DESAT_OFFSET · sin(π · (step - 60) / 60),  step ∈ [60, 120]
                = 0                                              otherwise
```

`DESAT_OFFSET = 5 %` (V3-03: reduced from 8 → recoverable challenge) produces
a half-sinusoidal depression of the `SpO2` equilibrium (peak at step 90). It
acts on the **infant SpO2 response model directly**, **not** as a sink on
`C_O2` (which is bounded below by 21 % and therefore could not produce a
measurable SpO2 dip under the old formulation). The new formulation gives a
measurable SpO2 dip (minimum ≈ 89.6 % under heater=0.25 + FiO2=0.40),
recoverable by raising FiO2. Feasibility: at peak, max FiO2=0.60 →
C_O2=56.45 → SpO2_eq = 92.32 % (inside [91,95] target band).

---

## 6. Episode Termination

Two distinct Gymnasium flags (the V2 review confirmed semantics):

* `terminated = True` when the agent **breaches** a hard safety bound:
  `Tc < 35.5 °C`, `Tc > 38.0 °C`, `SpO2 < 85 %`, or `SpO2 > 96 %`.
  The reward for that step includes the `r_safety = -100` penalty.
* `truncated = True` when `current_step >= max_steps` (200), i.e. the episode
  hits the 2.5-hour time budget without breach. Truncation is **not** a
  failure: the bootstrap target should use `V(s')` (not 0) when the episode
  ended via `truncated` (see Handoff §9).

In the v3.1 baseline, every one of the 25 evaluation episodes ran to
`step = 200` with **zero** safety violations across all five scenarios.

---

## 7. Discount Factor

The MDP is undiscounted-then-discounted: per-step reward is summed inside the
episode and the discount factor used in Bellman backups is

```
γ = 0.99   (config.yaml → training.gamma)
```

With 200-step episodes, the effective horizon `1/(1-γ) = 100` ≈ 75 min of
simulated time, which is short enough to keep the value function bounded yet
long enough to penalise any policy that drifts toward a hard safety breach.

---

## 8. MDP Tuple

Formally, the environment implements the tuple `M = (S, A, P, R, γ)`:

| Symbol | Instance |
|--------|----------|
| `S` | `Box(11,) float32` — the 11-row observation table in §2 |
| `A` | `Discrete(25)` — the 25-row action table in §3 |
| `P` | Stochastic transition kernel induced by the ODEs in §5 + Gaussian sensor noise (`σ_Tc = 0.01 °C`, `σ_SpO2 = 0.05 %`) |
| `R` | `R : S × A → ℝ` defined exactly by the weighted formula in §4 |
| `γ` | `0.99` |

The transition kernel `P` is **Markov** given the augmented observation that
includes `phase`, `disturbance` and `scenario_id` (see §9).

---

## 9. Design Assumptions (Markov property)

The environment is a **fully-observable** MDP under three augmentations
restored during the V2 review:

1. **`phase = step / max_steps`** (observation 8). Without normalised time, an
   identical state at step 10 and step 150 (e.g. on scenario 2) would have
   different next-state distributions because the ambient-temperature
   trajectory is time-dependent; `phase` resolves this.
2. **`disturbance ∈ {0, 1, 2}`** (observation 9). Exposes whether the
   porthole or the desaturation event is currently active, so the agent can
   distinguish "Tc dropped because of porthole" from "Tc dropped because of
   controller error".
3. **`scenario_id = (scenario − 1)/4 ∈ [0, 1]`** (observation 10, V2-04 fix).
   Two identical physical states from different scenarios (e.g. scenario 1
   vs scenario 5) had different future disturbance schedules under the old
   observation, violating the Markov property. The `scenario_id` feature
   exposes the hidden context and restores a fully-observable MDP.

The model further assumes: well-mixed incubator air, lumped-parameter
(no spatial gradients), single-baby thermal node, no humidity control,
preterm post-stabilisation physiology, no circulatory shock, and negligible
radiant-loss anisotropy.

---

## 10. Handoff Notes

* The `config.yaml` file is the **single source of truth** (V2-07): editing a
  scalar there overrides the corresponding module-level constant in
  `environment.py` at construction time via `load_config()`. No code change
  is needed to retune physics, reward weights, or safety bands.
* `OBS_NAMES` and `ACTION_TABLE` are frozen — Members 2 and 3 must use the
  exact same 11 feature names / 25 action indices.
* The reward formula in §4 is the **exact** implemented formula (V2-05);
  the worked example can be reproduced by running
  `tests/test_environment.py::test_reward_formula`.
* All V2 review issues (V2-01 to V2-08) are documented as resolved in
  `docs/handoff_to_member2.md` and `docs/parameter_provenance.md`.
* **Simulation parameters, not clinical guidance.** See
  `docs/safety_limits.md` and `docs/dataset_notes.md` for clinical sources.
