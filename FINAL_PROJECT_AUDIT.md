# Project 04 · Independent project audit

## Checklist revision · 8 November 2026

The revised submission now includes a rendered and validated **22-slide PPTX**, slide-by-slide notes, a checked related-work comparison, corrected oxygen references and a recorded actuator-switching plot. The old presentation plan is retired. A packaged Windows x64 / CPython 3.14 wheelhouse supplies the exact tested dependency lock through `reproducibility/install_offline.ps1`; the public index is not required by this route.

The new isolated environment passed all **95 tests** with **1,247 warnings**, no failures or skips (85.01 seconds). The Gymnasium checker passed. Fresh workflow, installation and independent-verification logs are in `reports/checklist_evidence/`. No application source, simulator configuration or checkpoint was modified in this revision.

Names supplied by the team: Ziad Kishk, Karim Ayman and Hassan Gebrill. Student IDs remain deferred at the user's request, so final administrative sign-off is pending. The historical training-configuration mismatch remains disclosed in the methods and presentation. Frozen inference/evaluation is reproducible; historical retraining is not claimed. Simulation only, not suitable for clinical deployment.

See `CHECKLIST_RESOLUTION.md` for item-by-item evidence and `presentation/Project04_Final_Presentation.pptx` for the final deck.


**Recommendation:** ready for an academic demonstration of the frozen-checkpoint benchmark. Historical training provenance has one unresolved discrepancy, documented below. The project is not suitable for clinical deployment.

## Evidence base

The incoming inventory contained 20 Python files, 60 checkpoints, 146 CSVs, 53 PNGs, 37 JSON files and the configuration, requirements and reports. SHA-256 records identify the incoming files. All 47 historical Member 01/02 plot files decoded successfully, and their 34 experimental JSON documents parsed. Detailed numerical and visual verification centered on the six current SHAP plots and the Member 03 decision/episode datasets. Historical training curves and sensitivity studies were retained as prior experiments. Training was not rerun. The project arrived without Git metadata.

Member 01’s Gymnasium environment resets to a configured initial state. Its step function decodes a joint heater/oxygen command, advances simplified physics and disturbances, computes four reward components and returns the next state with termination and truncation flags. The baseline uses core-temperature PID feedback and a rule for oxygen saturation. Member 02 normalizes 11 features, applies dense layers 11 → 128 ReLU → 128 ReLU → 25 linear estimates and selects a greedy action during evaluation. During historical learning, the online network selected the successor action, the target network valued it, and the replay buffer distinguished hard termination from time-limit truncation.

Member 03 records the decision state before applying the action. Both policies use the same episode seed. A shared explainer attributes the executed DDQN head to the observation features. Validation computes outcomes from the resulting state. The independent verifier replays the episodes and evaluates checkpoint arrays directly, without importing the Member 03 metric or attribution functions.

## Component inspection

| Files | Function and finding |
| --- | --- |
| src/environment.py, config.yaml | Gymnasium reset/step API, seeded noise, thermal/oxygen dynamics, parameter resolution and model thresholds. The default action table equals the per-instance table. |
| src/baseline.py | PID heater and saturation-driven oxygen rule; only Tc and SpO₂ are consumed from the 11-element input. |
| src/run_simulation.py, src/generate_plots.py, src/random_diagnostic.py, src/sensitivity_analysis.py | Member 01 trajectory export, plotting, random-policy diagnostics and one-parameter sweeps. Original data and plots retained separately from Member 03 evidence. |
| src/replay_buffer.py, src/q_network.py, src/ddqn_agent.py | Capacity-bounded replay; normalized dense online/target networks; DDQN targets, state restoration and inference. |
| src/train_ddqn.py, src/rl_utils.py, src/evaluate_controllers.py | Historical training/validation selection, seed partitions, termination-aware metrics and evaluation. The original trainer uses validation return for checkpoint selection. The evaluator was executed separately on the submitted checkpoint. |
| shap_explainer.py | Trained online inference, reference rollouts, SHAP adapters, fixed/greedy heads, contrast and six figure exports. Shape and reconstruction checks execute. |
| validation.py, run_member03.py | Full decision attribution, matched-seed trajectories, summary statistics and one shared explainer across stages. |
| tests/ | Original assertions preserved; three additional regressions cover error propagation, fallback and a full skip request. |
| tools/verify_submission.py | Independent NumPy model and DeepLIFT rescale calculation; direct Kernel calls; reference-state, trajectory, metric and manifest comparisons. |

## Contracts and findings

Feature order is Tc, Tsk, Ta, C_O2, SpO2, Tamb, dTa_dt, dC_O2_dt, phase, disturbance, scenario_id. Normalization uses checkpoint centers, scales and clipping. The 25 actions encode 5 × heater index + oxygen index. Heater fractions are 0, 0.25, 0.50, 0.75 and 1. Oxygen fractions are 0.21, 0.30, 0.40, 0.50 and 0.60. Decision-time small batches use the equivalent cached NumPy forward pass; Keras inference and serialized-array replay agreed within tolerance.

The selected `models/ddqn_tuned_seed0/best.pt` contains a trusted pickle tagged `neonatal-ddqn-tf-v1` with backend `tensorflow`. Both online and target weights, the normalizer and training metadata are stored. Its pre- and post-finalization SHA-256 is `823A2CA6988ADBF7DDB4A303DA82C1134BCFF0D18EF8C59AC21F52CC375411CE`.

The source sampler and independent rollout reconstruction both produced 82 idle, 8 porthole-active and 10 desaturation-active states over ticks 0–198. The sample is stratified and imbalanced. Coverage of early, late and disturbed states does not demonstrate population representativeness. Policy-level attribution pools selected heads. Only the contrast subtracts two heads at one state.

The incoming artifacts supplied only 120 SHAP-explained DDQN decisions and lacked episode keys in the transition CSVs. Kernel used the first 25 reference rows, excluding both active disturbances. Earlier charts could recalculate sampled explanations separately from the manifest, and validation could silently record attribution errors. Corrective changes, the source diff and the preserved original output bundle are listed in `MEMBER03_AUDIT_FIXES.md`.

For the final Deep output, the largest selected-head reconstruction difference across CSV rows was 1.70 × 10⁻⁶. The largest per-feature difference from independent rescale propagation was 4.18 × 10⁻⁷ or less. For Kernel the corresponding maxima were 1.53 × 10⁻⁷ and 1.40 × 10⁻⁷ or less. The verifier also checked all 25 global heads, the reference means, local example Q-values and the head-9 minus head-5 decomposition. These are numerical consistency results.

| Scenario | DDQN mean ± SD | Baseline mean ± SD | DDQN minus baseline |
| --- | --- | --- | --- |
| Normal | -25.63 ± 1.65 | -10.18 ± 0.81 | -15.44 |
| Ambient cooling | -18.47 ± 0.75 | -15.03 ± 1.75 | -3.44 |
| Porthole opening | -16.62 ± 0.98 | -11.75 ± 0.49 | -4.86 |
| Initial hypothermia | -16.87 ± 0.53 | -11.29 ± 0.68 | -5.57 |
| Oxygen desaturation | -25.71 ± 1.45 | -66.16 ± 0.89 | +40.45 |
| Overall | -20.66 ± 4.35 | -22.88 ± 22.16 | +2.23 |

The equal-weight mean advantage was +2.2269 points. The baseline won all 20 matched episodes in scenarios 1–4. The DDQN won all five desaturation pairs. This evaluation therefore favors different controllers as the frequency of the oxygen disturbance changes.

The greedy policy changed the joint action on 4,611 of 4,975 adjacent pairs (92.68%). The heater changed 3,782 times and oxygen changed 3,811 times, including 2,982 simultaneous changes. In 607 cases it returned immediately to the action from two decisions earlier. The independently replayed argmax matches every selected head. Exploration was disabled. The reward penalizes current squared actuator magnitude and has no explicit change-from-last-command term. Thus the winning Q head can change without a direct switching cost. Establishing how much that reward design caused the observed switching would require a separate controlled experiment.

## Reproducibility, provenance and limitations

The incoming and final `config.yaml` share SHA-256 `3962d9370508b1e4b9f0fcad8c2c128c796df1539422c2381ba58b973a041578`. The historical training and tuned evaluation manifests instead record `377e80775429788b88937786a6616d9d23f2b589c8bfe950ac0f2ed7b88e1973`. A simple LF/CRLF conversion does not explain the difference. Comparison of training metadata also shows legacy YAML keys and a different representation of the learning rate. These observations do not identify the exact historical edit. The corresponding older YAML bytes were not identified in the supplied project. All twelve historical source hashes match. The current configuration reproduces the 50 overlapping historical tuned evaluation returns and switch counts. The frozen-checkpoint benchmark is reproducible; byte-for-byte reconstruction of the original training configuration remains unresolved.

The prior root environment notes omitted the Keras nightly and transitive versions. The current lock contains all 58 active project distributions. Installation from the locally available artifacts and independent execution in a clean venv succeeded. Availability of those exact public-index binaries was not tested successfully on this host. The dependency and test reports document upstream warnings. No hardware-in-the-loop, external-patient-data or distribution-shift validation was performed.

This academic simulation is **not suitable for clinical deployment**. Its lumped thermal model, deterministic skin–core relationship and simplified chamber-oxygen and saturation dynamics limit physiological realism. The evaluation covers one checkpoint, five predefined scenarios and five paired seeds per scenario. Changes in patient physiology, sensor faults, actuator dynamics or combined disturbances can change the decisions and may produce unsafe actions outside the tested conditions.

The network supplies point estimates of Q without calibrated uncertainty. The DDQN receives 11 observations, including time, disturbance and scenario ID. The baseline reads core temperature and SpO₂ and maintains its own PID memory. This information asymmetry limits conclusions about the learning algorithm. Frequent switching raises an additional hardware question. The experiment measured simulated threshold avoidance. It supplies no patient-outcome evidence. SHAP establishes neither causality nor clinical safety.

## Final acceptance evidence

| Acceptance item | Evidence |
| --- | --- |
| Member 01 implementation, datasets and configuration preserved | Incoming file hashes, direct file comparison, seeded replay |
| Member 02 implementation and original checkpoint preserved | All twelve src files agree with the training manifest; before/after checkpoint hash matches |
| Explainer, background, feature order, Q head and action mapping | All-head tensor, independent 2-layer rescale and independent Kernel calculation |
| Global policy-head and fixed-head importance distinguished | Separate means and files, recomputed from 100 × 11 × 25 tensor |
| State, selected action, SHAP, reward and next state aligned | 10,000 replayed transitions, 5,000 explained DDQN decisions, keyed by episode seed |
| Normal, abnormal temperature and abnormal oxygen evaluated | All five named scenarios, five paired seeds per scenario and all three trajectory files |
| Baseline comparison and switching | Independent CSV reductions and separate Member 02 evaluator |
| Safety, scope and scientific language | Final reports and eight-slide notes explain modeled thresholds, information asymmetry and distribution shift |
| Six SHAP charts and validation outputs | Data-to-figure comparisons, local state replay, visual review and repeated file equality |
| CLI, skip behavior, failed Deep initialization and failure propagation | Four full commands, instrumented skip regression and real Kernel fallback test |
| Reproducibility, package versions and tests | 58-version lock, isolated pip check, 95 passed, compileall and clean demonstration |
| Synchronized reports and finalized presentation notes | Duplicate documents byte-compared during packaging |
| Final archive and reproduction instructions | Per-entry hash verification, ZIP CRC check, external ZIP digest and README commands |

