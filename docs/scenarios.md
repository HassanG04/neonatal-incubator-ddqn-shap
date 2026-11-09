# Scenarios — Neonatal Incubator v3.1

> This document describes the five canonical clinical scenarios used to
> stress-test the neonatal incubator environment. Each scenario has a fixed
> `scenario_id ∈ {1, 2, 3, 4, 5}` (mapped to the `scenario_id` observation
> feature), a fixed initial state, and a deterministic disturbance
> schedule. All numbers are simulation parameters, not clinical guidance.

The `scenario_id` feature (normalised `(scenario−1)/4 ∈ [0, 1]`) makes
the environment a fully-observable MDP — two states with identical
physical coordinates but different scenarios no longer alias, because
the agent can read the scenario from the observation.

---

## 1. Scenario 1 — `normal_standard`

* **SHAP class**: Normal
* **Clinical situation**: Stable preterm neonate, post-stabilisation, in a
  nominal incubator with no scheduled disturbance. The reference scenario —
  the controller must hold `Tc` near 37 °C and `SpO2` near 92 % for the full
  2.5 h episode. The target SpO2 band 91–95 % matches NICE guideline
  NG124 ("Specialist neonatal respiratory care for babies born preterm",
  recommendation 1.4.2) — V3-04 citation update.
* **Initial state**:

| Variable | Value | Unit |
|----------|------:|------|
| `Tc` | 37.0 | °C |
| `Tsk` | 36.5 | °C |
| `Ta` | 34.0 | °C |
| `C_O2` | 21.0 | % |
| `SpO2` | 92.0 | % |
| `Tamb` | 25.0 | °C |

* **Disturbance**: none.
* **Difficulty**: Easy. The PID baseline achieves mean `\|Tc − 37\| = 0.183 °C`,
  `SpO2 band_err = 0`, return = `−10.18 ± 0.23`.
* **What it tests**: Whether the agent can hold a stable setpoint without
  over-actuating. Useful for diagnosing reward-shaping bugs (e.g. energy
  term dominating).

---

## 2. Scenario 2 — `ambient_temperature_drop`

* **SHAP class**: Abnormal-temperature
* **Clinical situation**: A preterm neonate in a transport incubator or a
  NICU bay where the ambient room temperature drops gradually (e.g.
  nighttime AC setback, transport from warm delivery room to cooler
  NICU). The increasing heat loss forces the controller to raise
  `Q_heater` progressively.
* **Initial state**: identical to scenario 1 (`Tc=37, Ta=34, SpO2=92, Tamb=25`).
* **Disturbance math**:
  `Tamb(t) = 25.0 − 7.0 · (step / max_steps)` over 200 steps.
  At `step = 200`, `Tamb = 18 °C`. The ambient drop is **linear**, not
  abrupt.
* **Difficulty**: Moderate. Steady-state `Ta` at `u = 1.0`, `Tamb = 18`:
  `Ta = 18 + 360/8 = 63 °C` (well above setpoint) → controller has ample
  authority. Baseline achieves `return = −15.51 ± 1.61`, mean
  `\|Tc − 37\| = 0.215`.
* **What it tests**: Setpoint tracking under a slow, monotonic disturbance.
  SHAP should attribute the agent's heater decisions primarily to `Tamb`
  and `phase` features.

---

## 3. Scenario 3 — `porthole_opening_event`

* **SHAP class**: Abnormal-temperature
* **Clinical situation**: A clinician opens the incubator porthole for
  routine care (suctioning, line check) during steps [50, 80] (a 30-step
  ≈ 22.5 min window). The open porthole multiplies incubator heat loss by
  `PORTHOLE_H_C_FACTOR = 2.0`, producing a transient thermal perturbation.
  The incubator starts pre-warmed (`Ta = 36 °C`, not `34 °C`) — reflecting
  that porthole events happen *during* care, not at cold-start.
* **Initial state**:

| Variable | Value | Unit |
|----------|------:|------|
| `Tc` | 37.0 | °C |
| `Tsk` | 36.5 | °C |
| `Ta` | 36.0 | °C (pre-warmed) |
| `C_O2` | 21.0 | % |
| `SpO2` | 92.0 | % |
| `Tamb` | 25.0 | °C |

* **Disturbance math**:
  `h_c = 2.0` on `step ∈ [50, 80]`, `h_c = 1.0` elsewhere.
  During the event, the air-node loss term becomes
  `K_loss · (Ta − Tamb) · h_c = 8 · (Ta − 25) · 2 = 16·(Ta − 25) W`.
  The disturbance flag in the observation is `1.0` during the window,
  `0.0` elsewhere.
* **Feasibility**: At `u = 1.0`, `h_c = 2.0`, `Tamb = 25`, steady-state
  `Ta = 25 + 360/(8·2) = 47.5 °C` → the agent can hold `Tc ≥ 35.5`
  throughout.
* **Difficulty**: Hard. Baseline survives 200 steps with 0 violations and
  `return = −11.58 ± 0.86`; mean `\|Tc − 37\| = 0.191`.
* **What it tests**: Whether the agent can anticipate the disturbance (via
  the `disturbance = 1` feature) and pre-emptively raise `Q_heater`
  before `Tc` drops. SHAP should attribute heater decisions to
  `disturbance`, `Tamb`, and `dTa_dt` features during the window.

---

## 4. Scenario 4 — `initial_hypothermia`

* **SHAP class**: Abnormal-temperature
* **Clinical situation**: A preterm neonate admitted to the NICU with
  mild hypothermia (`Tc = 35.8 °C`, below the comfort band `36.5` but
  above the hard floor `35.5`) and mild desaturation (`SpO2 = 88 %`,
  below the target band `91–95` but above the hard floor `85`). The
  incubator air is pre-warmed (`Ta = 36 °C`) to allow the air-to-baby
  heat exchange to begin immediately. No time-windowed disturbance.
* **Initial state**:

| Variable | Value | Unit |
|----------|------:|------|
| `Tc` | 35.8 | °C |
| `Tsk` | 35.3 | °C |
| `Ta` | 36.0 | °C (pre-warmed) |
| `C_O2` | 21.0 | % |
| `SpO2` | 88.0 | % |
| `Tamb` | 25.0 | °C |

* **Disturbance**: none (initial-condition stress only).
* **Difficulty**: Moderate. The controller must rewarm the infant from
  `Tc = 35.8` to the setpoint `37.0` while also lifting `SpO2` from 88 to
  the `91–95` band. Baseline achieves `return = −11.11 ± 0.39`,
  mean `\|Tc − 37\| = 0.191`, `SpO2 band_err = 0.015` (residual leakage
  below band 91).
* **What it tests**: Recovery from an off-setpoint initial state without
  an ongoing disturbance. Useful for SHAP analysis of *transient* regimes
  (initial 20 steps differ markedly from the steady-state regime).

---

## 5. Scenario 5 — `oxygen_disturbance_desaturation` (V3-03 recoverable challenge)

* **SHAP class**: Abnormal-oxygen
* **Clinical situation**: A preterm neonate in nominal thermal
  conditions experiences an apnea-like desaturation event during steps
  [60, 120] (a 60-step ≈ 45 min window). The event **directly depresses
  the infant's `SpO2` equilibrium** by up to `DESAT_OFFSET = 5.0`
  percentage points (half-sinusoidal profile), modelling an intermittent
  physiological drive — **not** a leak in the incubator oxygen supply.
  The controller must raise `FiO2` to push `C_O2` up, thereby lifting
  `SpO2_eq` and partially offsetting the depression. Under V3-03 the
  disturbance is a **recoverable challenge**, not a stress test: at
  adequate FiO2 the [91, 95] target band is fully maintainable.
* **Initial state**: identical to scenario 1
  (`Tc=37, Ta=34, SpO2=92, Tamb=25`).
* **Disturbance math** (V3-03 fix):
  `desat_offset(t) = 5.0 · sin(π · (step − 60) / 60)`, `step ∈ [60, 120]`;
  `desat_offset = 0` elsewhere.
  `DESAT_OFFSET = 5.0 %` (reduced from 8.0 in v3.0), producing a peak
  depression at `step = 90` of 5 points on `SpO2_eq`. The `SpO2` state
  is lagged (`K_SPO2 = 0.02 1/s`) so the observed dip is smaller and
  slower than the equilibrium depression. The disturbance flag in the
  observation is `2.0` during the window.
* **Feasibility note** (V3-03, recoverable challenge):
  At peak disturbance (step 90), max `FiO2 = 0.60` gives
  `C_O2 = 56.45 %` → `SpO2_eq = 92 + 0.15·35.45 − 5 = 92.32 %` (inside
  the [91, 95] target band). Verified: with `heater = 0.25` +
  `FiO2 = 0.40`, min `SpO2` during the window = 89.6 % (just below the
  band, recoverable). With higher FiO2 the band is fully maintainable.
  Compare to v3.0 (`DESAT_OFFSET = 8.0`): the band was unreachable at
  any FiO2 and the baseline return was −246 (stress test).
* **Mechanistic note**: The disturbance acts on the **infant SpO2
  response model** (an apnea-like transient), **not** on `C_O2`. This is
  critical because `C_O2` is floored at 21 % (room air) and a sink on
  `C_O2` could not produce a measurable SpO2 dip. Under the V3-03
  formulation, the baseline achieves `SpO2 band_err = 0.323` (vs `0.000`
  for scenario 1), confirming the disturbance has a real but recoverable
  effect.
* **Difficulty**: Hard but recoverable. Baseline achieves
  `return = −66.82 ± 0.96`, mean `\|Tc − 37\| = 0.182`,
  `SpO2 band_err = 0.323`. All 5 baseline episodes survive to step 200
  with 0 safety violations (SpO2 minimum stays ≥ 85).
* **What it tests**: Whether the DDQN agent can *anticipate* the
  desaturation event (via the `disturbance = 2` feature) and raise
  `FiO2` *before* `SpO2` dips below the 91 lower band — and, with enough
  FiO2 authority, hold the band throughout the disturbance window. SHAP
  should attribute oxygen-action decisions primarily to `SpO2`, `C_O2`,
  `disturbance`, and `scenario_id` features.

---

## 6. Evaluation Protocol (10-Episode Default & 30-Episode Expanded Grid)

To produce statistically meaningful results across the five scenarios,
`src/evaluate_controllers.py` evaluates every controller (Baseline, DDQN seeds,
and standard DQN seeds) on identical held-out `(scenario, test_seed)` pairs
using `test_seed(scenario, i) = 900_000 + scenario*1000 + i` (strictly disjoint
from training and validation seeds):

* **Default held-out evaluation (`results/ddqn/`)**: **10 held-out test episodes per scenario** (`50` episodes per controller/seed, `350` episode records across 1 baseline + 3 DDQN seeds + 3 DQN seeds), run via:
  `PYTHONPATH=src python src/evaluate_controllers.py --episodes-per-scenario 10`
* **Expanded held-out evaluation (`results/ddqn_30ep/`)**: **30 held-out test episodes per scenario** (`150` episodes per controller/seed, `1,050` episode records across all controllers), run via:
  `PYTHONPATH=src python src/evaluate_controllers.py --episodes-per-scenario 30 --output-dir results/ddqn_30ep --plots-dir plots/ddqn_30ep`
  *(Note: Historical Member 1 handoff notes proposed a 30-episode `{0..4} × {10..14}` cross-seed grid; both the standard 10-episode/scenario and expanded 30-episode/scenario held-out evaluations are executed and archived for full traceability.)*
* **Reporting**: for each scenario, report mean ± standard deviation of
  * total episode reward
  * `mean |Tc − 37|` (°C)
  * `mean SpO2 band_err` (squared deviation outside [91, 95])
  * number of safety violations (target: 0)
  * episode length (target: 200 steps)
* **Comparison**: head-to-head against the Member 1 rule-based baseline
  (numbers in `docs/handoff_to_member2.md §6`). The DDQN agent must
  beat the baseline on **scenario 5** (return > −66.82) and match on
  scenarios 1–4 (return > −16) to be considered an improvement. Under
  V3-03, the head-room on scenario 5 is smaller than in v3.0 (when the
  target was −246), so Member 2 should focus on anticipatory FiO2
  control rather than reactive.
* **Cross-scenario training**: recommended single DDQN across all 5
  scenarios, leveraging the `scenario_id` feature; per-scenario training
  is an acceptable alternative if the cross-scenario agent
  under-performs.
* **Ablation hook (V3-01)**: pass `config_path="ablation.yaml"` to
  `NeonatalIncubatorEnv(...)` to override any parameter (including
  `desat_offset`, `q_heater_max`, `k_loss`, all reward weights, all
  safety bands) without modifying `environment.py`. Useful for testing
  `desat_offset ∈ {3, 5, 7}` sensitivity curves (V3-03 ablation).

### Reproducibility checklist

1. `config.yaml` loaded (single source of truth, V3-01).
2. `python -c "import yaml; yaml.safe_load(open('config.yaml'))"` succeeds.
3. `PYTHONPATH=src python -m environment` runs the smoke test for all 5
   scenarios with seed 42.
4. `results/summary.csv` regenerated from `python src/run_simulation.py`
   matches the table in `docs/handoff_to_member2.md §6` to within seed
   noise.
5. The `scenario_id` feature in the observation equals `(scenario − 1)/4`
   for every step (contract test).
6. `env.desat_offset == 5.0` after construction with the default config
   (V3-03 regression check).
7. `env.action_table` is a list of 25 dicts and `env.action_table[24]["Q_heater_W"]`
   equals `env.q_heater_max` (V3-01 instance-level action table check).
8. No "NG122" occurrences in any `docs/*.md` file (V3-04 citation check).

---

**Simulation parameters, not clinical guidance.**
