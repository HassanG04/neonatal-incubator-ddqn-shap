# Handoff to Member 2 — Environment v3.1 (all V3 issues fixed)

> **From**: Member 1 (environment + rule-based baseline)
> **To**: Member 2 (DDQN training)
> **Date**: v3.1 release
> **Scope**: This handoff documents the *complete* environment contract that
> Member 2 builds on. Every value below is mirrored in `config.yaml` — the
> **single source of truth** (V3-01). All numbers are simulation parameters,
> not clinical guidance.

---

## 1. Summary of V3 fixes (all 5 resolved)

The V3 review raised 5 issues against the v3.0 environment. All 5 are
resolved in v3.1:

| ID | Issue (V3 review) | Resolution in v3.1 |
|----|--------------------|---------------------|
| **V3-01** | `load_config()` mutated module globals; `__init__` did not store all params on the instance; `action_table` was a module-level constant that ignored `q_heater_max` overrides. | `load_config()` now returns a **resolved dict** (no module-global mutation). `__init__` stores ALL params on the instance: `k_loss, k_baby, c_air, c_core, porthole_h_c_factor, k_blend, k_leak, desat_offset, k_spo2, spo2_base, spo2_gain, spo2_min, spo2_max, tc_safe_low/high, spo2_safe_low/high, w_thermal, w_oxygen, w_energy, w_safety, r_safety_penalty, tc_setpoint, spo2_target_low/high`. `action_table` is rebuilt **per-instance** from `self.q_heater_max`. A custom `config_path` argument can override ALL of these for ablation studies. |
| **V3-02** | `C_AIR` and `C_CORE` were labelled as raw air/infant heat capacities, but the numerical values (1500 and 2000 J/°C) were calibrated *effective* masses. | Both parameters are now documented as **effective thermal masses** in `docs/parameter_provenance.md`: `C_AIR` represents the effective thermal mass of the incubator's air + walls + mattress + internal structure (a 30 L air volume alone would be ~36 J/°C, but the effective mass including structure is ~1500 J/°C). `C_CORE` represents the effective core thermal mass of the infant's central compartment (brain + thoracic organs + circulating blood), calibrated to give a ~500 s core time constant matching observed neonatal thermal response. Both are labelled **"calibrated educational parameter"** in the provenance. |
| **V3-03** | Scenario 5 with `DESAT_OFFSET = 8.0` produced a stress test (mean reward −246, SpO2 dip below 85 hard floor on some seeds). The desaturation disturbance was too severe to be a recoverable challenge. | `DESAT_OFFSET` reduced from 8.0 → **5.0**. Scenario 5 is now a **RECOVERABLE CHALLENGE** (not a stress test). Feasibility verified at peak disturbance (step 90): max `FiO2 = 0.60` → `C_O2 = 56.45 %` → `SpO2_eq = 92 + 0.15·35.45 − 5 = 92.32 %` (inside [91, 95] target band). With `heater = 0.25` + `FiO2 = 0.40`, min `SpO2` during window = 89.6 % (just below band, recoverable). With higher `FiO2` the band is maintainable. |
| **V3-04** | "NICE NG122" citations were stale; the correct guideline for preterm respiratory care is NG124. | All "NICE NG122" references replaced with **"NICE NG124"** ("Specialist neonatal respiratory care for babies born preterm", recommendation 1.4.2). Affected files: `docs/scenarios.md`, `docs/dataset_notes.md`, `docs/safety_limits.md`, `docs/parameter_provenance.md`, `docs/presentation_notes.md`. |
| **V3-05** | Stale "10-feature" references remained in `baseline.py` and `run_simulation.py`; handoff listed test functions that no longer existed. | Stale "10-feature" references removed from `baseline.py` and `run_simulation.py`. The handoff test list below now lists **only test functions that exist** in `tests/test_environment.py` and `tests/test_baseline.py`. |

---

## 2. Action mapping (Discrete(25) = 5 × 5)

`action = heater_idx × 5 + o2_idx`.

| heater_idx | o2_idx | heater_level | fio2_command | Q_heater (W) |
|------------|--------|--------------|--------------|-------------|
| 0 | 0–4 | 0.00 | 0.21 / 0.30 / 0.40 / 0.50 / 0.60 | 0 |
| 1 | 0–4 | 0.25 | 0.21 / 0.30 / 0.40 / 0.50 / 0.60 | 90 |
| 2 | 0–4 | 0.50 | 0.21 / 0.30 / 0.40 / 0.50 / 0.60 | 180 |
| 3 | 0–4 | 0.75 | 0.21 / 0.30 / 0.40 / 0.50 / 0.60 | 270 |
| 4 | 0–4 | 1.00 | 0.21 / 0.30 / 0.40 / 0.50 / 0.60 | 360 |

`fio2_command` is the **absolute** volume fraction delivered to the blender
(`0.21 = 21 %`, room air; `0.60 = 60 %`). It is not a normalised `[0, 1]`
fraction. The oxygen mass-balance uses `FiO2·100` to convert to percent.

Note (V3-01): the `Q_heater (W)` column above is computed from
`self.q_heater_max = 360.0`. If Member 2 supplies a custom `config_path`
that overrides `q_heater_max`, the per-instance `action_table` is rebuilt
from the new value automatically — the Q-network must re-read the new
mapping at construction time.

---

## 3. Observation (11-dim, frozen feature order)

| # | Feature | Unit | Range | Description |
|---|---------|------|-------|-------------|
| 0 | `Tc` | °C | [30, 42] | Core body temperature |
| 1 | `Tsk` | °C | [30, 42] | Skin temperature (`= Tc − 0.5`) |
| 2 | `Ta` | °C | [20, 40] | Incubator air temperature |
| 3 | `C_O2` | % | [21, 60] | Inspired O2 concentration |
| 4 | `SpO2` | % | [70, 100] | Arterial O2 saturation (separate lagged state) |
| 5 | `Tamb` | °C | [15, 35] | Ambient room temperature |
| 6 | `dTa_dt` | °C/s | [−2, +2] | Realized rate of `Ta` |
| 7 | `dC_O2_dt` | %/s | [−3, +3] | Realized rate of `C_O2` |
| 8 | `phase` | − | [0, 1] | Normalised episode time `step / 200` |
| 9 | `disturbance` | − | {0, 1, 2} | 0 idle, 1 porthole, 2 O2 desaturation |
| 10 | `scenario_id` | − | [0, 1] | `(scenario − 1)/4` (restores Markov) |

The Q-network input layer must be `nn.Linear(11, hidden)` — *not* 10.
The 11 feature names are available as `environment.OBS_NAMES`.

---

## 4. Reward formula — exact and worked

```
r_thermal = -(Tc - 37.0)²

band_err_sq(SpO2, [91, 95]) =
    0                     if SpO2 ∈ [91, 95]
    (91 - SpO2)²          if SpO2 < 91
    (SpO2 - 95)²          if SpO2 > 95
r_oxygen  = -band_err_sq(SpO2, [91, 95])

u_fio2_norm = (fio2_cmd - 0.21) / 0.39            # ∈ [0, 1]
r_energy  = -(u_heater² + u_fio2_norm²)

r_safety  = -100  if Tc ∉ [35.5, 38.0] OR SpO2 ∉ [85, 96]
         =    0   otherwise

R = 1.0·r_thermal + 0.5·r_oxygen + 0.05·r_energy + 1.0·r_safety
```

**Worked example** (`Tc = 37.0`, `SpO2 = 93.0`, `u_heater = 0.5`, `fio2 = 0.30`, safe):

| term | value |
|------|------:|
| `r_thermal` | `0` |
| `r_oxygen` | `0` (in band) |
| `u_fio2_norm` | `0.231` |
| `r_energy` | `-(0.25 + 0.053) = -0.303` |
| `r_safety` | `0` |
| `R` | `0.05 · (-0.303) = -0.0152` |

All four weights live in `config.yaml → reward.weights` and are stored on the
environment instance as `env.w_thermal`, `env.w_oxygen`, `env.w_energy`,
`env.w_safety` (V3-01). They may be overridden at construction time by
supplying a custom `config_path`.

---

## 5. Termination vs Truncation (with bootstrap rule)

| Flag | Trigger | Bootstrap rule |
|------|---------|----------------|
| `terminated = True` | Hard safety breach: `Tc ∉ [35.5, 38.0]` OR `SpO2 ∉ [85, 96]` (reward includes `r_safety = -100`). | `target = r` — **terminal**, do *not* bootstrap `V(s')`. |
| `truncated = True` | `current_step ≥ 200` (2.5 h simulated time) — episode budget exhausted. | `target = r + γ · Q(s', argmax_a Q(s', a; θ_online); θ_target)` — **bootstrap normally** (truncation is not a failure). |

The standard Gymnasium convention applies. A common bug is to zero the
bootstrap on `truncated` episodes; Member 2 must guard against it
(`(1 - terminated)` is the correct mask, not `(1 - done)`).

---

## 6. Baseline performance (v4, 5 episodes/seed=42, authoritative)

Numbers below come from `results/summary.csv` and `results/trajectory_*.csv`
produced by `python src/run_simulation.py`. They are **reproducible**.

| Scenario | Mean total_reward | Mean \|Tc − 37\| (°C) | Mean SpO2 band_err | Safety violations | Steps survived |
|----------|------------------:|----------------------:|-------------------:|-------------------:|---------------:|
| 1 normal_standard | **−10.18 ± 0.23** | 0.183 | 0.000 | 0 | 200 |
| 2 ambient_temperature_drop | **−15.51 ± 1.61** | 0.215 | 0.000 | 0 | 200 |
| 3 porthole_opening_event | **−11.58 ± 0.86** | 0.191 | 0.000 | 0 | 200 |
| 4 initial_hypothermia | **−11.11 ± 0.39** | 0.191 | 0.015 | 0 | 200 |
| 5 oxygen_disturbance_desaturation | **−66.82 ± 0.96** | 0.182 | 0.323 | 0 | 200 |

### Key finding for Member 2

* The corrected baseline (FF_BIAS=0.25) stabilizes all 5 scenarios
  with zero safety violations and 200-step episodes. Mean Tc tracks
  within 0.18-0.22 °C of the 37.0 °C target.
* Scenarios 1-4 produce similar returns (≈ −10 to −16) — the baseline
  PID controller is well-tuned there, so DDQN must beat −15 to be
  useful.
* **Scenario 5 is the recoverable challenge** the DDQN must beat. It
  shows the clearest effect: SpO2 band_err = 0.323 (vs 0.000 for
  scenarios 1-3), reward −66.82 (vs −10 to −15). The disturbance still
  produces a measurable SpO2 dip (peak depression of 5 points on
  `SpO2_eq` at step 90), but the band is recoverable: at peak
  disturbance with `FiO2 = 0.60`, `SpO2_eq = 92.32 %` (inside the
  [91, 95] target band). This is the *head-room* the DDQN agent should
  exploit.
* Scenario 3 remains feasible (200 steps survived, 0 violations).
* All 5 evaluation episodes ran to `step = 200` with **0 safety
  violations** — the simulator is now safe to train on without frequent
  early terminations.

### Config override note (V3-01)

Every value in the table above is reproduced from the default
`config.yaml`. Member 2 may pass `config_path="ablation.yaml"` to
`NeonatalIncubatorEnv(...)` to override **any** parameter — including
`desat_offset`, `q_heater_max`, `k_loss`, all reward weights, and all
safety bands — without modifying `environment.py`. This makes ablation
studies (e.g. varying `desat_offset` between 3, 5, 7) trivial to script.
The resolved config dict is also exposed as `env._config` for inspection.

---

## 7. Validated DDQN Setup (`config.yaml → training` & Hyperparameter Study)

Hyperparameters below live in `config.yaml → training` and were selected using **validation seeds only** (`validation_seed(scn, i) = 500_000 + scn*1000 + i`, 15 validation episodes per evaluation) via the controlled 10-run ablation study recorded in `results/ddqn/hyperparameter_study.csv`:

| Hyperparameter | Original v1 Value | Tuned v2 Value (`config.yaml`) | Notes / Validation Rationale |
|----------------|-------------------|--------------------------------|------------------------------|
| Discount `γ` | `0.99` | `0.99` | `1/(1−γ) = 100` steps ≈ 75 min horizon |
| Learning rate (`lr`) | `3.0e-4` | **`1.0e-4`** | Prevents post-40k Q-value overestimation drift; improved late-stage validation return on seed 0 from `-82.45` to `-42.31` (100% survival) |
| Batch size | `128` | `128` | Mini-batch sampled from uniform replay buffer |
| Replay buffer (`buffer_size`) | `100 000` | `100 000` | Stores `(s, a, r, s', terminated, truncated, scenario)` |
| Warmup steps | `3 000` | `3 000` | Initial exploration steps before gradient updates |
| Total steps | `100 000` | `100 000` | Per training seed (`train_freq = 2`, `48 500` gradient steps) |
| Target update | `hard` every `500` grad steps | `hard` every `500` grad steps | (`250`, `1000`, and `soft τ=0.005` also evaluated in `hyperparameter_study.csv`) |
| Reward scale | `0.1` | `0.1` | Stabilizes Huber (`δ = 1.0`) TD targets and global grad clip (`10.0`) |
| `ε_start` → `ε_end` | `1.0` → `0.02` | **`1.0` → `0.01`** | Lower final `ε` halves late random-action hyperoxia terminations |
| `ε_decay_steps` | `40 000` | **`20 000`** | Linear decay after warmup; best single-factor validation return (`-22.19`, last-5 mean `-31.00`, 100% survival) |
| `baseline_guided_prob` | `0.0` | `0.0` | Pure RL exploration (`0.15` guided ablation tested in `hyperparameter_study.csv`, which introduced bias and degraded peak validation return to `-33.88`) |
| Seeds | `[0, 1, 2]` | `[0, 1, 2]` | Original runs preserved in `models/ddqn_seed{0,1,2}/`; tuned runs saved in `models/ddqn_tuned_seed{0,1,2}/` |
| Evaluation protocol | **10 eps / scenario** (`50` eps/model) | **10 eps / scenario** (`results/ddqn_tuned/`) + **30 eps / scenario** (`results/ddqn_30ep/`) | Held-out `test_seed(scn, i) = 900_000 + scn*1000 + i` (never used for tuning) |
| Q-network & Checkpoint | `Dense(128, relu) × 2 → Dense(25, linear)` | TensorFlow/Keras (`neonatal-ddqn-tf-v1`) | Saved with `ObservationNormalizer` via Python `pickle` in `.pt` files; load trusted files via `DDQNAgent.load()`, **never** `torch.load()` |

### Suggested reward-shaping experiments

* Increase `w_oxygen` to 1.0 if the agent under-attends to scenario 5.
  (Set via custom `config_path`, V3-01.)
* Increase `w_safety` to 5.0 if early-training episodes breach frequently.
* Add an `r_progress = +0.01` for surviving each step (curiosity bonus) —
  optional, document it if used.
* Vary `desat_offset` ∈ {3.0, 5.0, 7.0} across runs to map the
  disturbance–return sensitivity curve (V3-03 ablation).

---

## 8. What Member 2 returns to Member 3

Member 2's deliverable to Member 3 (SHAP / XAI) must include:

1. **11 frozen feature names** (the exact `OBS_NAMES` tuple, in order).
2. **Trained DDQN weights** (`models/ddqn_<scenario>.pt` or one consolidated
   `models/ddqn.pt` if a single agent is trained across all scenarios).
3. **A `q_values(obs: np.ndarray) -> np.ndarray` interface** returning the
   `(25,)` Q-value vector for any 11-dim observation.
4. **A `predict(obs) -> (action, q_values)` interface** for deterministic
   evaluation.
5. **Training logs**: episode returns, losses, ε schedule, hyperparameters
   used (so Member 3 can stratify SHAP analysis by training phase).
6. **Evaluation table** in the same format as §6 (5 scenarios × 5 seeds),
   so Member 1's baseline and Member 2's DDQN can be compared directly.

The 11 feature names Member 3 must use for SHAP attribution are, in order:

```
["Tc", "Tsk", "Ta", "C_O2", "SpO2", "Tamb",
 "dTa_dt", "dC_O2_dt", "phase", "disturbance", "scenario_id"]
```

---

## 9. How to run / verify (exact commands)

```bash
# 1. Verify config.yaml loads (V3-01 single source of truth)
python -c "import yaml; yaml.safe_load(open('config.yaml')); print('OK')"

# 2. Run the environment smoke test (5 scenarios, seed=42)
cd /home/z/my-project/download/member_1
PYTHONPATH=src python -m environment

# 3. Run the rule-based baseline (5 episodes -> results/summary.csv)
PYTHONPATH=src python src/run_simulation.py --episodes 5 --seed 42

# 4. Run the unit tests (tests/test_environment.py, tests/test_baseline.py)
PYTHONPATH=src python -m pytest tests/ -v

# 5. (Member 2) Train DDQN & DQN agents (original v1 preserved; tuned v2 under distinct names)
PYTHONPATH=src python src/train_ddqn.py --smoke
for s in 0 1 2; do
  PYTHONPATH=src python src/train_ddqn.py --seed $s --algo ddqn --run-name ddqn_tuned_seed$s --set lr=1e-4 eps_decay_steps=20000 eps_end=0.01
  PYTHONPATH=src python src/train_ddqn.py --seed $s --algo dqn  --run-name dqn_tuned_seed$s  --set lr=1e-4 eps_decay_steps=20000 eps_end=0.01
done

# 6. (Member 2) Evaluate original v1 runs (10 eps/scenario) and tuned v2 runs (10 and 30 eps/scenario)
PYTHONPATH=src python src/evaluate_controllers.py --episodes-per-scenario 10 --output-dir results/ddqn --plots-dir plots
PYTHONPATH=src python src/evaluate_controllers.py \
  --ddqn-runs ddqn_tuned_seed0 ddqn_tuned_seed1 ddqn_tuned_seed2 \
  --dqn-runs dqn_tuned_seed0 dqn_tuned_seed1 dqn_tuned_seed2 \
  --episodes-per-scenario 10 --output-dir results/ddqn_tuned --plots-dir plots/ddqn_tuned
PYTHONPATH=src python src/evaluate_controllers.py \
  --ddqn-runs ddqn_tuned_seed0 ddqn_tuned_seed1 ddqn_tuned_seed2 \
  --dqn-runs dqn_tuned_seed0 dqn_tuned_seed1 dqn_tuned_seed2 \
  --episodes-per-scenario 30 --output-dir results/ddqn_30ep --plots-dir plots/ddqn_30ep
```

The `PYTHONPATH=src` export is required because `environment.py` lives under
`src/` and the `config.yaml` loader expects to find it at `../config.yaml`
relative to the environment module.

---

## 10. Acceptance criteria (all 5 V3 issues documented as fixed)

| ID | Acceptance criterion | Where verified |
|----|----------------------|----------------|
| V3-01 | `load_config()` returns a resolved dict; `__init__` stores all 24 params on the instance; `action_table` rebuilt from `self.q_heater_max`; `config_path` overrides all params. | `tests/test_environment.py::test_config_resolved_dict`, `test_instance_params`, `test_action_table_per_instance`; `src/environment.py:421–495` |
| V3-02 | `C_AIR = 1500` labelled "effective incubator thermal mass (air + walls + mattress + internal structure)"; `C_CORE = 2000` labelled "effective infant core thermal mass (brain + thoracic organs + central blood volume)"; both type = "calibrated educational parameter". | `docs/parameter_provenance.md §1` (C_AIR and C_CORE rows) |
| V3-03 | `DESAT_OFFSET = 5.0`; scenario 5 return is −66.82 ± 0.96 (v4 corrected, FF_BIAS=0.25; was −73.34 ± 2.29 in v3.1, −246 in v3.0); SpO2 band_err = 0.323; feasibility note (step-90 peak `SpO2_eq = 92.32 %`) documented. | `config.yaml:86`, `results/summary.csv` (scn 5 rows), `docs/scenarios.md §5` |
| V3-04 | Zero occurrences of "NG122" across all docs; "NG124" appears with the recommendation 1.4.2 citation in `dataset_notes.md`, `safety_limits.md`, `scenarios.md`, `parameter_provenance.md`, `presentation_notes.md`. | `rg -i "NG122" docs/` returns no matches |
| V3-05 | No "10-feature" or "OBS_DIM=10" references in `baseline.py` or `run_simulation.py`; the test-function list below contains only existing test names. | `rg "10-feature\|OBS_DIM.*10" src/baseline.py src/run_simulation.py` returns no matches |

---

## 11. Files you need (Member 2's working set)

| Path | Purpose |
|------|---------|
| `config.yaml` | Single source of truth (V3-01). Override any parameter via a custom `config_path` — do not edit the YAML values you cannot justify in `docs/parameter_provenance.md`. |
| `src/environment.py` | The MDP — do not modify; consume via `NeonatalIncubatorEnv`. All parameters are stored on the instance (V3-01). |
| `src/baseline.py` | Reference rule-based controller. Reuse `HEATER_LEVELS`, `FIO2_LEVELS` constants for action compatibility. |
| `src/run_simulation.py` | Reproduces the baseline numbers in §6. |
| `tests/test_environment.py` | Contract tests — run before training and after every code change. |
| `docs/mdp_specification.md` | Normative MDP spec. |
| `docs/api_reference.md` | Programmer's reference. |
| `docs/scenarios.md` | Scenario descriptions + evaluation protocol. |
| `docs/safety_limits.md` | Safety band sources (clinical literature). |
| `docs/parameter_provenance.md` | Parameter sources (V3-02 effective thermal masses). |
| `docs/dataset_notes.md` | Dataset / parameter justification. |

---

## 12. Existing test functions (V3-05 cleanup)

The following test functions exist in `tests/test_environment.py` and
`tests/test_baseline.py` and are the only ones referenced by this handoff
(stale references removed, V3-05):

| File | Test function |
|------|---------------|
| `tests/test_environment.py` | `test_obs_dim_is_eleven` |
| `tests/test_environment.py` | `test_obs_in_bounds` |
| `tests/test_environment.py` | `test_scenario_id_feature` |
| `tests/test_environment.py` | `test_reward_formula` |
| `tests/test_environment.py` | `test_terminated_truncated_mask` |
| `tests/test_environment.py` | `test_scenario5_desat_dip` |
| `tests/test_environment.py` | `test_porthole_feasibility` |
| `tests/test_environment.py` | `test_config_resolved_dict` (V3-01) |
| `tests/test_environment.py` | `test_instance_params` (V3-01) |
| `tests/test_environment.py` | `test_action_table_per_instance` (V3-01) |
| `tests/test_environment.py` | `test_desat_offset_is_5` (V3-03) |
| `tests/test_baseline.py` | `test_baseline_returns_match_summary` |
| `tests/test_baseline.py` | `test_action_table_compat` |

If Member 2 needs additional tests, add them to these two files; do not
edit the existing test names.

---

## 13. Open issues for Member 2 (not blocking)

* The reward `r_thermal = -(Tc − 37)²` saturates at `Tc = 35.5` (penalty ≈ −2.25
  per step). If the agent struggles to leave the hypothermia region, consider
  adding a "comfort-band bonus" `+0.5·𝟙[36.5 ≤ Tc ≤ 37.5]`. Document any such
  shaping in the deliverable report.
* The replay buffer is currently uniform. PER (Prioritized Experience Replay)
  may help on scenario 5 — but use the same buffer for all scenarios so the
  comparison stays fair.
* Cross-scenario training (one DDQN on all 5 scenarios) vs per-scenario
  training (5 DDQN agents) is an open question. Recommend cross-scenario
  training with the `scenario_id` feature, evaluated against per-scenario.
* With `DESAT_OFFSET = 5.0` (V3-03), scenario 5 is now a recoverable
  challenge. The DDQN agent's head-room is smaller than in v3.0 (when the
  disturbance was 8.0) but more clinically plausible: the agent must
  *anticipate* the desaturation, not just survive it.

---

## 14. Risk register (transferred to Member 2)

| Risk | Mitigation |
|------|------------|
| Agent exploits reward shaping instead of true safety. | Evaluate on `safety_violations` (must stay 0) and `mean \|Tc − 37\|` (must drop below 0.183). |
| `ε-decay` too fast → poor exploration on scenario 5. | Use `ε_decay = 0.997` if scenario-5 return plateaus. |
| Target network divergence. | Use soft update `τ = 0.005` instead of hard sync every 10 episodes. |
| Replay buffer too small for 5 scenarios × 500 episodes. | Increase to `200 000` if loss curves diverge. |
| V3-01 `config_path` override silently changes environment behaviour between ablations. | Always log `env._config` at the start of each run; compare against the default `config.yaml` before training. |

---

## 15. Sign-off / Contact

Member 1 has verified the v3.1 environment against all 5 V3 acceptance
criteria (§10). The baseline numbers in §6 are reproducible from
`config.yaml` + `src/environment.py` + `src/baseline.py` +
`src/run_simulation.py`. Member 2 should be able to start DDQN training
immediately with no further changes to the environment.

For any ambiguity, the canonical reference is `config.yaml` and
`src/environment.py` — both kept in sync (V3-01). If they disagree, the
YAML wins (the Python constants are defaults only). File any discrepancy
as a GitHub issue and tag Member 1.

**Simulation parameters, not clinical guidance.**


---

## 16. V5 Final acceptance — DDQN smoke test (V5-06)

Member 2 can verify the environment is ready for training by running this
minimal DDQN smoke test (no Member 1 code changes required):

```python
# ddqn_smoke_test.py — Member 2 verification script
import numpy as np
from environment import NeonatalIncubatorEnv, OBS_NAMES, ACTION_TABLE

env = NeonatalIncubatorEnv(scenario=1, seed=42)
obs, info = env.reset(seed=42)
assert obs.shape == (11,), f"obs shape {obs.shape} != (11,)"
assert env.action_space.n == 25
assert len(OBS_NAMES) == 11
assert len(ACTION_TABLE) == 25

# Smoke run: random actions, verify API contract
total_reward = 0.0
terminated = truncated = False
steps = 0
while not (terminated or truncated):
    action = np.random.randint(25)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    # Verify observation always in bounds
    assert env.observation_space.contains(obs), f"obs out of bounds at step {steps}"
    # Verify reward components sum correctly
    computed = (info['w_thermal'] * info['r_thermal']
                + info['w_oxygen'] * info['r_oxygen']
                + info['w_energy'] * info['r_energy']
                + info['w_safety'] * info['r_safety'])
    assert abs(computed - info['reward']) < 1e-6, "reward formula mismatch"

print(f"DDQN smoke test PASS: {steps} steps, reward={total_reward:.2f}")
print(f"  terminated={info['terminated']}, truncated={info['truncated']}")
print(f"  termination_cause available in info: safe={info['safe']}")
print("  Environment is ready for DDQN training.")
```

### V5 acceptance evidence

| # | Check | Evidence |
|---|-------|----------|
| V5-01 | No stale baseline numbers in docs | `grep -rn '\-13.07\|\-16.44\|\-73.34\|0.235\|0.239' docs/ README.md` → 0 active-context matches |
| V5-02 | Sensitivity analysis documented | `docs/parameter_provenance.md §3.6` + `results/sensitivity_results.csv` (45 rows) + 3 plots |
| V5-02 | O2 equation matches code | `docs/presentation_notes.md` Slide 6: `dSpO2/dt = K_SPO2·(SpO2_eq - SpO2)`, offset applied once in `SpO2_eq` |
| V5-03 | Custom action grids supported | `src/baseline.py::_rule_fio2` uses dynamic index selection (closest-to-target); tested 3×2, 5×5, 2×3, 4×2 grids — all pass, no IndexError |
| V5-04 | Random diagnostic script | `src/random_diagnostic.py` + `results/random_diagnostic.csv` (100 rows) + `plots/random_diagnostic_episode_length.png` |
| V5-05 | Real Gymnasium check_env | gymnasium 1.4.0; `check_env` passes on all 5 scenarios; 62/62 tests pass |
| V5-06 | DDQN smoke test | Script above runs without Member 1 code changes; API contract verified |

### Environment / dependency versions (V5-05: record for reproducibility)

```
python:        3.12.3
gymnasium:     1.4.0
tensorflow:    2.17.0
numpy:         1.26.4
pandas:        2.2.2
matplotlib:    3.10.8
pyyaml:        6.0.2
pytest:        9.2.0
config.yaml:   sha256 = <compute with: python -c "import hashlib; print(hashlib.sha256(open('config.yaml','rb').read()).hexdigest())">
seed plan:     base_seed + scenario*1000 + episode (base=42); disjoint RL train/val/test seeds in src/rl_utils.py
```

**The environment and DDQN/DQN training & evaluation pipeline are complete and verified.**

---

## 17. Post-Handoff Investigation & Resolution Report (Problems 1–5)

*(Simulation-only academic benchmark — not clinically validated or suitable for real neonatal care.)*

### 17.1 Problem 1 — Seed Consistency & Checkpoint Selection
* **Root Cause of Original v1 Variance**: Under `lr=3e-4, eps_decay_steps=40000, eps_end=0.02` (`models/ddqn_seed{0,1,2}/`, preserved), Q-values drifted upward after step 40,000, causing `ddqn_seed0` and `ddqn_seed1` to peak early on validation (`step = 15,000`, `grad_steps = 6,001`, held-out test returns `−31.62` and `−35.93`) while `ddqn_seed2` peaked at `step = 85,000` (`−17.28`).
* **Best vs Final Checkpoints (v1)**: Selecting `best.pt` on validation seeds outperformed `final.pt` on 5 of 6 v1 runs (`ddqn_seed0`: `−31.62` vs `−100.63`; `ddqn_seed2`: `−17.28` vs `−32.55`).
* **Validation-Selected Fix (`v2_tuned`: `lr=1e-4, eps_decay_steps=20000, eps_end=0.01`)**:
  - Trained across seeds `0, 1, 2` under unique names (`models/ddqn_tuned_seed{0,1,2}/` and `models/dqn_tuned_seed{0,1,2}/`).
  - All 3 DDQN seeds converge late (`best_val_step = 100k, 90k, 100k` with validation returns `−20.07, −19.44, −19.65`).
  - On the 10-episode/scenario held-out test set (`results/ddqn_tuned/eval_aggregate.csv`), **every single DDQN seed beats the rule-based baseline (`−22.87`)**: `seed0 = −20.47`, `seed1 = −19.63`, `seed2 = −19.44` (**combined DDQN mean = `−19.85 ± 5.49`**, inter-seed std reduced from `7.97` to **`0.45`**, survival = **100.0%**).
  - On the expanded 30-episode/scenario held-out test set (`results/ddqn_30ep/eval_aggregate.csv`), **every single DDQN seed again beats the baseline (`−23.12`)**: `seed0 = −20.25`, `seed1 = −19.76`, `seed2 = −19.41` (**combined DDQN mean = `−19.81 ± 5.59`**, inter-seed std = **`0.34`**, survival = **100.0%** across 450 episodes).

### 17.2 Problem 2 — Early Safety Terminations During Training & Denominator Precision
* **Denominator Effect**: In v1 (`eps_decay_steps=40000`), the "last 20% of completed episodes by episode count" spans steps `~30,000–100,000` because early exploratory episodes are ~8 steps long while late surviving episodes are 200 steps long. Measuring the true **last 20% of environment steps (`steps ∈ [80,000, 100,000]`)** yields a much smaller episode termination rate (`32.0%–39.6%` in v1, and **`10.5%–13.0%` in v2 tuned**), and a per-environment-step termination rate of only **`0.06%–0.07%` per step** in v2 (`results/ddqn/training_termination_diagnostics.csv`).
* **Exploratory vs Greedy Action Attribution**: `src/train_ddqn.py` logs `random_actions`, `greedy_actions`, `guided_actions`, and `last_action_source` per episode. In late training (`steps ≥ 80,000`) of the tuned DDQN runs, **33 of 38 terminated episodes (`86.8%`)** were terminated on an exploratory (`ε = 0.01`) random action step (primarily random `FiO2 ≥ 0.40` pushing `SpO2 > 96%` hyperoxia when the policy operates near `94–95%`), whereas the greedy (`ε = 0`) policy achieves **100.0% survival** across all held-out test episodes.

### 17.3 Problem 3 — Scenario-Level Trade-Off Analysis (Scenarios 1–4 vs Scenario 5)
* **Why Baseline Wins Scenarios 1, 3, and 4**: The rule-based controller has a continuous PID loop + exact room-air default (`FiO2 = 0.21`), giving `SpO2 = 92.0%` (`spo2_band_err = 0.000`) and low control effort (`0.28–0.36`). By contrast, the single multi-scenario DDQN policy maintains a slightly elevated `FiO2` (`~34–38%`, `SpO2 ≈ 93.8–94.5%`) and higher discrete switching (`effort ≈ 0.66–0.77`), incurring `~5–9` points of cumulative energy and minor `SpO2 > 95%` band penalty over 200 steps.
* **Scenario 2 (Ambient Drop)**: Tuned DDQN matches the baseline within `0.84` reward points on the 30-episode evaluation (`−16.47` vs `−15.63`) and actually achieves **lower mean core temperature error** (`0.168 °C` vs baseline `0.215 °C`), trading slightly higher heater/FiO2 energy for tighter thermal tracking.
* **Scenario 5 (Oxygen Desaturation)**: Tuned DDQN outperforms the baseline on **100% of held-out episodes** (`−27.98 ± 3.82` vs `−66.69 ± 0.99`, paired gain **+38.71 ± 1.78** points on 30-ep), cutting `SpO2` band error by more than **70%** (`0.095%` vs `0.325%`).

### 17.4 Problem 4 & 5 — Controlled Hyperparameter Study & Reproducibility Verification
* **Controlled Study**: All 16 runs (10 single-factor seed-0 ablations + 3 tuned DDQN seeds + 3 tuned DQN seeds) are recorded in `results/ddqn/hyperparameter_study.csv` with exact overrides, validation metrics, and termination diagnostics.
* **Verification**: All 79 unit/integration tests (`pytest tests/ -v`) and `gymnasium.utils.env_checker.check_env()` pass on real `gymnasium 1.4.0` and `tensorflow 2.17.0`.
