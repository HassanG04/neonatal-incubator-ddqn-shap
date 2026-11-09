# Dataset Notes — Neonatal Incubator v3.1

> This document justifies the parameter values used in the v3.1 neonatal
> incubator environment and points to the clinical literature that
> motivated them. All values are **simulation parameters** tuned for an
> educational RL benchmark, not clinical guidance. The v3.1 release
> introduces V3-02 (effective thermal masses for `C_AIR` and `C_CORE`),
> V3-03 (recoverable scenario 5 with `DESAT_OFFSET = 5.0`), and V3-04
> (NICE NG124 citation replacing NG122).

---

## 1. Population scope

The simulator models a **preterm post-stabilisation** neonate (NICE NG124
terminology, recommendation 1.4.2 — V3-04), i.e. an infant who has been
admitted to the NICU, has had immediate resuscitation completed, and is
now in the incubator for ongoing thermal and respiratory support. The
physiological target bands are:

* Axillary / core temperature `Tc` ∈ `[36.5, 37.5] °C` (normothermia, AAP);
  modeled termination limits `Tc < 35.5 °C` or `Tc > 38.0 °C`
  under the simulator rule.
* Pre-ductal SpO2 ∈ `[91, 95] %` (NICE NG124 preterm post-stabilisation,
  recommendation 1.4.2 — V3-04); modeled termination thresholds `SpO2 < 85 %`
  or `SpO2 > 96 %` under the modeled termination rule.

The simulator does **not** model: extreme prematurity (< 28 weeks),
congenital heart disease, circulatory shock, post-surgical recovery, or
therapeutic hypothermia for HIE.

---

## 2. Parameter justification table (v3.1 values)

| Parameter | Value | Unit | Why this value |
|-----------|------:|------|----------------|
| `DT_STEP` | 45.0 | s | Long enough for an RL control step (typical NICU nurse action interval), short enough that sub-stepping at `N_SUBSTEPS = 10` keeps `k · dt_sub < 0.5` (stability). |
| `MAX_STEPS` | 200 | − | `200 × 45 s = 9000 s = 2.5 h` — one nursing shift handoff window. |
| `N_SUBSTEPS` | 10 | − | `dt_sub = 4.5 s`. For the fastest rate `K_BLEND = 0.05 1/s`, `k·dt_sub = 0.225 ≪ 0.5` (stable). |
| `K_LOSS` | 8.0 | W/°C | Well-insulated double-wall incubator. At full heating (`Q = 360 W`, `Tamb = 25`): steady-state `Ta = 25 + 360/8 = 70 °C` (sufficient authority). During porthole (`h_c = 2.0`): `Ta = 25 + 360/(8·2) = 47.5 °C` (still feasible). |
| `K_BABY` | 4.0 | W/°C | Lumped baby↔air conductance; chosen so the core time constant `C_CORE / K_BABY = 500 s ≈ 11 control steps` (slow physiological response). |
| `C_AIR` | 1500.0 | J/°C | **V3-02 effective incubator thermal mass** (calibrated educational parameter). Represents the effective mass of the incubator's air + walls + mattress + internal structure. A 30 L air volume alone is ~36 J/°C (`30 L · 1.2 kg/m³ · 1005 J/(kg·K)`), but the effective mass including the double-wall acrylic structure, mattress, humidifier reservoir, and internal tray is calibrated to ~1500 J/°C — yielding an air time constant `C_AIR / (K_LOSS·h_c + K_BABY) = 1500 / (16 + 4) = 75 s` (porthole) or `1500 / 12 = 125 s` (closed), both inside the literature 60–180 s range. |
| `C_CORE` | 2000.0 | J/°C | **V3-02 effective infant core thermal mass** (calibrated educational parameter). Represents the effective mass of the infant's central compartment (brain + thoracic organs + circulating blood). A whole-body 2 kg infant at 3500 J/(kg·°C) = 7000 J/°C, but the *core* that responds to air temperature is smaller (~600 g effective), giving ~2000 J/°C. Calibrated so the core time constant `C_CORE / K_BABY = 500 s` matches observed neonatal thermal response (literature: 5–15 min). |
| `PORTHOLE_H_C_FACTOR` | 2.0 | − | Calibrated for porthole feasibility. At `h_c = 3.5`, full heating could only reach `Ta = 32 °C` (infeasible, baseline terminated). At `h_c = 2.0`, `Ta = 47.5 °C` (feasible). |
| `K_BLEND` | 0.05 | 1/s | Blender + inhalation mixing. Time constant 20 s (realistic for a premature infant ventilator circuit). |
| `K_LEAK` | 0.005 | 1/s | Slow leak toward room air (21 %). Time constant 200 s ≈ 4 control steps. |
| `K_SPO2` | 0.02 | 1/s | SpO2 lag response. Time constant 50 s (slower than blender, faster than core temperature). |
| `SPO2_BASE` | 92.0 | % | SpO2 equilibrium at `FiO2 = 0.21` (room air). Matches the typical SpO2 of a healthy preterm neonate at 36–37 °C body temperature. |
| `SPO2_GAIN` | 0.15 | %/% | Lifting `C_O2` from 21 to 40 % raises `SpO2_eq` by `0.15 × 19 = 2.85` points (92 → 94.85). Within the target band, so the agent has authority to compensate the 5-point `desat_offset` in scenario 5 (V3-03). |
| `DESAT_OFFSET` | **5.0** | % | **V3-03 fix** (reduced from 8.0): peak `SpO2_eq` depression during scenario 5's [60, 120] window. Now a *recoverable challenge*: at peak (step 90), max `FiO2 = 0.60` gives `C_O2 = 56.45 %` → `SpO2_eq = 92 + 0.15·35.45 − 5 = 92.32 %` (inside [91, 95] target band). With `heater = 0.25` + `FiO2 = 0.40`, min SpO2 = 89.6 % (just below band, recoverable). Baseline `SpO2 band_err` drops from 0.687 (v3.0) to **0.323** (v3.1). |
| `Q_HEATER_MAX` | 360.0 | W | Maximum heater power. Typical double-wall incubator: 200–400 W heater. |
| `TC_SAFE_LOW / HIGH` | 35.5 / 38.0 | °C | Conservative inside the moderate-hypothermia floor and the hyperthermia threshold. |
| `SPO2_SAFE_LOW / HIGH` | 85 / 96 | % | Educational simulation termination limits, not validated clinical alarm thresholds. |
| `SPO2_TARGET_LOW / HIGH` | 91 / 95 | % | NICE NG124 preterm post-stabilisation band (recommendation 1.4.2) — V3-04 citation. |
| `R_SAFETY_PENALTY` | 100.0 | − | Large enough to dominate any per-step reward (~order −1 to −5). |

---

## 3. V3-02 / V3-08 parameter provenance table

For every physics / reward / safety parameter, the table below lists its
**source type** (measured / literature-derived / assumed / calibrated /
calibrated educational parameter). The full per-parameter provenance
including sensitivity notes is in `docs/parameter_provenance.md`.

| Parameter | Units | Value | Source | Type |
|-----------|-------|------:|--------|------|
| `DT_STEP` | s | 45.0 | Engineering choice (NICU control interval) | Assumed |
| `MAX_STEPS` | − | 200 | 2.5-h nursing shift window | Assumed |
| `N_SUBSTEPS` | − | 10 | Numerical stability (`k·dt_sub < 0.5`) | Calibrated |
| `K_LOSS` | W/°C | 8.0 | Double-wall incubator heat-loss coefficient | Literature-derived |
| `K_BABY` | W/°C | 4.0 | Lumped baby↔air conductance | Assumed |
| `C_AIR` | J/°C | 1500.0 | **Effective incubator thermal mass** (air + walls + mattress + internal structure); calibrated to ~75 s air time constant | **Calibrated educational parameter** (V3-02) |
| `C_CORE` | J/°C | 2000.0 | **Effective infant core thermal mass** (brain + thoracic organs + central blood volume); calibrated to ~500 s core time constant | **Calibrated educational parameter** (V3-02) |
| `PORTHOLE_H_C_FACTOR` | − | 2.0 | Porthole open vs closed heat loss ratio | Calibrated |
| `K_BLEND` | 1/s | 0.05 | Ventilator circuit mixing rate | Assumed |
| `K_LEAK` | 1/s | 0.005 | Slow leak toward room air | Assumed |
| `K_SPO2` | 1/s | 0.02 | SpO2 lag time constant 50 s | Literature-derived |
| `SPO2_BASE` | % | 92.0 | Healthy preterm SpO2 at room air | Literature-derived |
| `SPO2_GAIN` | %/% | 0.15 | Slope of SpO2-vs-FiO2 curve in mid-range | Assumed |
| `DESAT_OFFSET` | % | **5.0** | Apnea-like SpO2 depression magnitude (V3-03 recoverable challenge) | **Calibrated educational parameter** (V3-03) |
| `Q_HEATER_MAX` | W | 360.0 | Typical double-wall incubator heater | Literature-derived |
| `TC_SAFE_LOW/HIGH` | °C | 35.5/38.0 | WHO hypothermia classification; AAP normothermia band | Literature-derived |
| `SPO2_SAFE_LOW/HIGH` | % | 85/96 | Educational termination limits | Assumed |
| `SPO2_TARGET_LOW/HIGH` | % | 91/95 | **NICE NG124** preterm post-stabilisation band, recommendation 1.4.2 (V3-04) | Literature-derived |
| `R_SAFETY_PENALTY` | − | 100.0 | Engineering choice (dominate per-step reward) | Assumed |
| Reward weights `w_thermal, w_oxygen, w_energy, w_safety` | − | 1.0, 0.5, 0.05, 1.0 | Multi-objective trade-off | Calibrated |

---

## 4. Dataset generation procedure

The "dataset" used by Members 2 and 3 consists of:

1. **State-action trajectories** from `src/run_simulation.py`
   (`results/trajectory_scenario{N}_ep{K}.csv`, one file per episode,
   25 episodes total in the v3.1 baseline).
2. **Aggregate metrics** in `results/summary.csv` (one row per episode).
3. **Trained DDQN & DQN weights** (`models/ddqn_seed{0,1,2}/` and
   `models/dqn_seed{0,1,2}/`, TensorFlow/Keras checkpoints serialized via
   Python `pickle` with embedded `ObservationNormalizer`; load via
   `DDQNAgent.load()`, not `torch.load()`) evaluated on the held-out
   test grid (**10 episodes/scenario = 50 episodes/model** in
   `results/ddqn/`, plus the expanded **30 episodes/scenario = 150
   episodes/model** in `results/ddqn_30ep/`).

### Reproducibility

* Master seed `42` (`config.yaml → environment.seed`) for the 25-episode rule-based baseline (`results/summary.csv`).
* Disjoint RL seed partition (`src/rl_utils.py`):
  - Training seeds (`--seed 0, 1, 2`): `train_seed(run_seed, ep_idx) = 1_000_000 * (run_seed + 1) + ep_idx`.
  - Validation seeds (used for hyperparameter & `best.pt` selection only): `validation_seed(scenario, i) = 500_000 + scenario * 1000 + i`.
  - Held-out test seeds (touched only by `src/evaluate_controllers.py`): `test_seed(scenario, i) = 900_000 + scenario * 1000 + i`.
* All trajectories are byte-reproducible from `config.yaml` +
  `src/environment.py` + `src/baseline.py` + `src/run_simulation.py`.
* V3-01 ablation hook: Member 2 may pass `config_path="ablation.yaml"`
  to `NeonatalIncubatorEnv(...)` to override any parameter without
  modifying `environment.py`. The resolved config dict is exposed as
  `env._config` for inspection.

---

## 5. Known limitations of the dataset

1. **Synthetic**: All trajectories come from the lumped-parameter ODE
   model, not from real NICU data. Clinical validation against
   retrospective EHR data is out of scope for this RL+XAI benchmark.
2. **Single infant phenotype**: One lumped-parameter infant;
   inter-patient variability not modelled.
3. **Deterministic disturbances**: Scenario 2 (ambient drop), 3
   (porthole), and 5 (desaturation) follow fixed schedules. Stochastic
   disturbance arrivals are a possible extension.
4. **No sensor faults**: Probe-off, motion artefact, and low-perfusion
   SpO2 artefacts are not modelled.
5. **No circadian effects**: No day/night variability in `Tamb`, no
   diurnal SpO2 patterns.
6. **Effective thermal masses** (V3-02): `C_AIR` and `C_CORE` are
   *calibrated effective* parameters, not direct measurements. They
   capture the lumped response of the incubator structure and the
   infant's core compartment respectively, but do not model spatial
   gradients (e.g. head-vs-foot temperature, radiant-vs-convective
   asymmetry).
7. **Recoverable scenario 5** (V3-03): the desaturation disturbance is
   calibrated to be recoverable at adequate FiO2. Real apnea-of-prematurity
   events are often more severe and may require bag-mask ventilation
   rather than FiO2 adjustment alone.

---

## References and clinical scope

The authoritative checked bibliography is [RELATED_WORK.md](RELATED_WORK.md). NICE NG124 recommendation 1.4.2 motivates the post-stabilization 91-95% target. The SUPPORT oxygen-targeting paper is *Target Ranges of Oxygen Saturation in Extremely Preterm Infants*, DOI [10.1056/NEJMoa0911781](https://doi.org/10.1056/NEJMoa0911781). Its comparison of 85-89% and 91-95% target ranges does not establish the simulator's 85/96% abort thresholds.

Earlier documents incorrectly used the CPAP/surfactant article (DOI 10.1056/NEJMoa0911783, PMID 20472939) to support oxygen limits. That reference and the associated claims are superseded. Historical documents remain in `reports/checklist_evidence/superseded_docs.zip`. This correction changes documentation only.

## 7. Update log (v2.x → v3.0 → v3.1)

| Parameter | v2.x value | v3.0 value | v3.1 value | Reason |
|-----------|-----------:|----------:|----------:|--------|
| `K_LOSS` | 15 | 8 | 8 | V2-02: porthole feasibility (unchanged in v3.1) |
| `C_AIR` | 200 | 1500 | 1500 | V2-02: realistic air time constant; V3-02: relabelled as *effective* incubator thermal mass (calibrated educational parameter) |
| `C_CORE` | 2000 | 2000 | 2000 | V3-02: relabelled as *effective* infant core thermal mass (calibrated educational parameter) |
| `PORTHOLE_H_C_FACTOR` | 3.5 | 2.0 | 2.0 | V2-02: feasibility (unchanged in v3.1) |
| `DESAT_OFFSET` | (sink on C_O2) | 8 (on SpO2_eq) | **5 (on SpO2_eq)** | V2-01: physiological model; V3-03: recoverable challenge (return −66.82, band_err 0.323) |
| `OBS_DIM` | 10 | 11 | 11 | V2-04: +`scenario_id` (unchanged in v3.1) |
| `dTa_dt, dC_O2_dt` | pre-step ODE estimates | realized post-clipping differences | realized post-clipping differences | V2-03: bounds-correct (unchanged in v3.1) |
| Reward `r_oxygen` | linear error | squared band error | squared band error | V2-05: sharper near band edges (unchanged in v3.1) |
| Parameter source table | absent | present (V2-08) | present (V2-08 + V3-02 effective masses + V3-03 recoverable DESAT) | Provenance audit, refined |
| NICE citation | n/a | NG122 (stale) | **NG124**, recommendation 1.4.2 | V3-04: corrected citation |
| Config loading | module-global mutation | module-global mutation | **resolved dict + instance-level params + per-instance action_table** | V3-01: single source of truth + ablation hook |

### v3.1 baseline performance (5 episodes, seed=42)

| Scenario | Mean total_reward | Mean \|Tc-37\| (°C) | Mean SpO2 band_err | Safety violations | Steps |
|----------|------------------:|---------------------:|-------------------:|-------------------:|------:|
| 1 | −10.18 ± 0.23 | 0.183 | 0.000 | 0 | 200 |
| 2 | −15.51 ± 1.61 | 0.215 | 0.000 | 0 | 200 |
| 3 | −11.58 ± 0.86 | 0.191 | 0.000 | 0 | 200 |
| 4 | −11.11 ± 0.39 | 0.191 | 0.015 | 0 | 200 |
| 5 | −66.82 ± 0.96 | 0.182 | 0.323 | 0 | 200 |

**Simulation parameters, not clinical guidance.**

