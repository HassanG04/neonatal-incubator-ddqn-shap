# Project 04 · Final handover

## Checklist revision · 8 November 2026

The revised submission now includes a rendered and validated **22-slide PPTX**, slide-by-slide notes, a checked related-work comparison, corrected oxygen references and a recorded actuator-switching plot. The old presentation plan is retired. A packaged Windows x64 / CPython 3.14 wheelhouse supplies the exact tested dependency lock through `reproducibility/install_offline.ps1`; the public index is not required by this route.

The new isolated environment passed all **95 tests** with **1,247 warnings**, no failures or skips (85.01 seconds). The Gymnasium checker passed. Fresh workflow, installation and independent-verification logs are in `reports/checklist_evidence/`. No application source, simulator configuration or checkpoint was modified in this revision.

Names supplied by the team: Ziad Kishk, Karim Ayman and Hassan Gebrill. Student IDs remain deferred at the user's request, so final administrative sign-off is pending. The historical training-configuration mismatch remains disclosed in the methods and presentation. Frozen inference/evaluation is reproducible; historical retraining is not claimed. Simulation only, not suitable for clinical deployment.

See `CHECKLIST_RESOLUTION.md` for item-by-item evidence and `presentation/Project04_Final_Presentation.pptx` for the final deck.


## A. Submission decision

**Ready for academic submission and a simulation-only demonstration of the frozen checkpoint.** All requested runtime modes completed, both SHAP methods were checked per feature, and the validation data reproduced independently. Historical training-file provenance is partially resolved, as described in section I. No patient or hardware validation was performed.

## B. Member responsibilities

| Member | Implemented contribution | Independent evidence |
| --- | --- | --- |
| 01 | Gymnasium simulator, simulation-generated datasets, MDP and PID/rule baseline | Seeded replay reproduced every next state, reward and baseline action. Source, configuration and original outputs were preserved. |
| 02 | DDQN, observation scaling, replay buffer, training and evaluation | Frozen checkpoint format and tensor shapes verified. Member 02’s evaluator reproduced all 50 individual returns and switch counts. |
| 03 | SHAP explainers, local and global views, decision validation and integration | Two complete SHAP workflows, six inspected figures, corrected full decision logging, independent feature-level checks and 95 passing tests. |

## C. Architecture

Member 01’s Gymnasium environment resets to a configured initial state. Its step function decodes a joint heater/oxygen command, advances simplified physics and disturbances, computes four reward components and returns the next state with termination and truncation flags. The baseline uses core-temperature PID feedback and a rule for oxygen saturation. Member 02 normalizes 11 features, applies dense layers 11 → 128 ReLU → 128 ReLU → 25 linear estimates and selects a greedy action during evaluation. During historical learning, the online network selected the successor action, the target network valued it, and the replay buffer distinguished hard termination from time-limit truncation.

Member 03 records the decision state before applying the action. Both policies use the same episode seed. A shared explainer attributes the executed DDQN head to the observation features. Validation computes outcomes from the resulting state. The independent verifier replays the episodes and evaluates checkpoint arrays directly, without importing the Member 03 metric or attribution functions.

## D. Experimental findings

| Scenario | DDQN mean ± SD | Baseline mean ± SD | DDQN minus baseline |
| --- | --- | --- | --- |
| Normal | -25.63 ± 1.65 | -10.18 ± 0.81 | -15.44 |
| Ambient cooling | -18.47 ± 0.75 | -15.03 ± 1.75 | -3.44 |
| Porthole opening | -16.62 ± 0.98 | -11.75 ± 0.49 | -4.86 |
| Initial hypothermia | -16.87 ± 0.53 | -11.29 ± 0.68 | -5.57 |
| Oxygen desaturation | -25.71 ± 1.45 | -66.16 ± 0.89 | +40.45 |
| Overall | -20.66 ± 4.35 | -22.88 ± 22.16 | +2.23 |

The equal-weight mean advantage was +2.2269 points. The baseline won all 20 matched episodes in scenarios 1–4. The DDQN won all five desaturation pairs. This evaluation therefore favors different controllers as the frequency of the oxygen disturbance changes.

| Metric | DDQN | Baseline |
| --- | --- | --- |
| Mean absolute core-temperature error (°C) | 0.197 | 0.191 |
| Core temperature in band (%) | 95.72 | 98.72 |
| Mean SpO₂ band error (percentage points) | 0.055 | 0.065 |
| SpO₂ in band (%) | 81.62 | 95.26 |
| Mean combined effort (normalized units) | 0.749 | 0.318 |
| Heater energy (kJ) | 1111.97 | 984.64 |
| Joint-action switches per episode | 184.44 | 120.12 |
| Modeled hard-threshold terminations | 0 | 0 |

The greedy policy changed the joint action on 4,611 of 4,975 adjacent pairs (92.68%). The heater changed 3,782 times and oxygen changed 3,811 times, including 2,982 simultaneous changes. In 607 cases it returned immediately to the action from two decisions earlier. The independently replayed argmax matches every selected head. Exploration was disabled. The reward penalizes current squared actuator magnitude and has no explicit change-from-last-command term. Thus the winning Q head can change without a direct switching cost. Establishing how much that reward design caused the observed switching would require a separate controlled experiment.

## E. SHAP findings

Phase had the largest mean absolute attribution across the selected heads (0.3921) and to fixed head 22 (0.4060). The reference set had 100 states, including eight active porthole and ten active desaturation states. At oxygen-event tick 90, head 9 exceeded head 5 by 0.1030 Q units at the same heater level. Its largest positive feature differences were SpO₂ (+0.8359) and chamber oxygen (+0.6512). Positive features offset a negative difference in background values. These are model estimates relative to a deliberately sampled background.

| Scenario | Decision tick | Head | Heater / FiO₂ fractions | Background mean Q | Actual Q |
| --- | --- | --- | --- | --- | --- |
| Normal | 50 | 2 | 0.00 / 0.40 | 1.0467 | 1.1417 |
| Ambient cooling | 150 | 5 | 0.25 / 0.21 | 1.0096 | 0.1722 |
| Porthole opening | 65 | 1 | 0.00 / 0.30 | 1.0208 | 1.2279 |
| Oxygen desaturation | 90 | 9 | 0.25 / 0.60 | -0.1853 | 2.5336 |

## F. Execution evidence

The incoming 92-test result reproduced. The final clean suite collected 95 and passed all 95, with 1,247 warnings, no failures and no skips. Compilation and the subsequent integrated demo exited 0. Default, explicit Deep and the isolated demo reproduced all 14 delivered files byte for byte. Skip and Kernel yielded identical episode metrics. Deep and Kernel decision contributions agreed with separate computations within 4.18 × 10⁻⁷ and 1.40 × 10⁻⁷ respectively. See `reports/FINAL_TEST_RESULTS.md` for timings and warning categories.

## G. Reproduction

Run from the extracted project root on Windows x64 with CPython 3.14.6. If the Windows Python launcher is unavailable, use the full path to that interpreter for the first command.

```powershell
./reproducibility/install_offline.ps1 -Python python -Environment .venv
./.venv/Scripts/Activate.ps1
python -m pip check
python -m pytest tests/ -v
python run_member03.py
```

The supplied offline installer is the supported path. See `reproducibility/README.md` for manual commands and wheel provenance. The public-index command could not resolve the lock on this host. The interpreter itself is required separately.


Use separate destinations to retain the submitted Deep SHAP results alongside a new run.

```powershell
python shap_explainer.py --method deep --output-dir outputs/reproduction/shap
python validation.py --shap-method deep --output-dir outputs/reproduction/validation
python run_member03.py --skip-shap --output-dir outputs/validation_only
python run_member03.py --shap-method deep --output-dir outputs/deep_repeat
python run_member03.py --shap-method kernel --output-dir outputs/kernel
python tools/verify_submission.py --output-dir outputs/member03
```

The background seed defaults to 42. Evaluation seeds are fixed independently at 900000 + 1000 × scenario + episode index and pair the two controllers. The exact CLI commands without output overrides were also exercised in separate project copies. Skip mode updates validation only. Existing figures can remain from an earlier run if a directory is reused. Consult the current validation JSON for attribution status.

## H. Deliverables

- **Source and tests.** `src/` contains twelve original Member 01/02 modules. Member 03 entry points are `shap_explainer.py`, `validation.py` and `run_member03.py`. `tools/verify_submission.py` independently checks results. `tests/` contains the original five files plus one new regression module.
- **Models and history.** `models/ddqn_tuned_seed0/best.pt` is the submitted checkpoint. The other checkpoints, Member 01/02 data and figures remain under `models/`, `models_v1_reference/`, `results/` and `plots/`. `outputs/member03_incoming.zip` preserves the original Member 03 output files.
- **Six figures.** `shap_summary.png`, `shap_feature_importance.png`, `shap_policy_feature_importance.png`, `shap_waterfall.png`, `shap_action_contrast.png` and `shap_scenario_comparison.png` sit under `outputs/member03/shap/` with `shap_manifest.json`, `background_metadata.json` and `global_attributions.npz`.
- **Validation data.** `normal_scenario_results.csv`, `abnormal_temperature_results.csv`, `abnormal_oxygen_results.csv`, `validation_episodes.csv` and `validation_summary.json` are under `outputs/member03/validation/`. Scenario recovery diagnostics and transcripts are under `reports/final_evidence/`.
- **Reports and archive.** Root-level reports cover scientific validation, SHAP, the project audit, changes, progress, acceptance, environment, speaking notes and handover. `reports/FINAL_TEST_RESULTS.md` documents execution. `reports/SUBMISSION_FILES.sha256` enumerates individual paths including historical datasets. The distributed ZIP has a separate SHA-256 sidecar.

## I. Limitations and open historical question

The incoming and final `config.yaml` share SHA-256 `3962d9370508b1e4b9f0fcad8c2c128c796df1539422c2381ba58b973a041578`. The historical training and tuned evaluation manifests instead record `377e80775429788b88937786a6616d9d23f2b589c8bfe950ac0f2ed7b88e1973`. A simple LF/CRLF conversion does not explain the difference. Comparison of training metadata also shows legacy YAML keys and a different representation of the learning rate. These observations do not identify the exact historical edit. The corresponding older YAML bytes were not identified in the supplied project. All twelve historical source hashes match. The current configuration reproduces the 50 overlapping historical tuned evaluation returns and switch counts. The frozen-checkpoint benchmark is reproducible; byte-for-byte reconstruction of the original training configuration remains unresolved.

This academic simulation is **not suitable for clinical deployment**. Its lumped thermal model, deterministic skin–core relationship and simplified chamber-oxygen and saturation dynamics limit physiological realism. The evaluation covers one checkpoint, five predefined scenarios and five paired seeds per scenario. Changes in patient physiology, sensor faults, actuator dynamics or combined disturbances can change the decisions and may produce unsafe actions outside the tested conditions.

The network supplies point estimates of Q without calibrated uncertainty. The DDQN receives 11 observations, including time, disturbance and scenario ID. The baseline reads core temperature and SpO₂ and maintains its own PID memory. This information asymmetry limits conclusions about the learning algorithm. Frequent switching raises an additional hardware question. The experiment measured simulated threshold avoidance. It supplies no patient-outcome evidence. SHAP establishes neither causality nor clinical safety.

The locked isolation test used a local wheelhouse with cached and reconstructed wheels. A fresh download of the full lock from the public package index was not verified. The tested paths are Windows x64 and CPU.

## J. Recommendation for the committee

Demonstrate the four CLI modes or the default run followed by the independent verifier. Defend the desaturation result alongside the other four scenarios, both target-band metrics and switching. The evidence supports the reported behavior of this checkpoint on the stated simulator and seed set. Recreating historical training or assessing deployment would require additional evidence.

