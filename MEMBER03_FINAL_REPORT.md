# Member 03 final technical report

Member 03 integrated the fixed online DDQN, the baseline, the paired-seed evaluator and two SHAP methods. Independent seeded replay verified every decision and transition. The incoming sparse set of 120 DDQN explanations is preserved separately. The final dataset explains all 5,000 DDQN decisions.

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

## What the explanations show

| Scenario | Decision tick | Head | Heater / FiO₂ fractions | Background mean Q | Actual Q |
| --- | --- | --- | --- | --- | --- |
| Normal | 50 | 2 | 0.00 / 0.40 | 1.0467 | 1.1417 |
| Ambient cooling | 150 | 5 | 0.25 / 0.21 | 1.0096 | 0.1722 |
| Porthole opening | 65 | 1 | 0.00 / 0.30 | 1.0208 | 1.2279 |
| Oxygen desaturation | 90 | 9 | 0.25 / 0.60 | -0.1853 | 2.5336 |

For the 100 reference states, phase had the largest selected-head mean absolute attribution (0.3921), followed by disturbance (0.1481) and scenario identity (0.1211). For head 22, phase (0.4060) and disturbance (0.1636) were followed by ambient temperature (0.1407). Individual signed profiles describe specific decisions. The ranking can change with the chosen head, state or background.

The greedy policy changed the joint action on 4,611 of 4,975 adjacent pairs (92.68%). The heater changed 3,782 times and oxygen changed 3,811 times, including 2,982 simultaneous changes. In 607 cases it returned immediately to the action from two decisions earlier. The independently replayed argmax matches every selected head. Exploration was disabled. The reward penalizes current squared actuator magnitude and has no explicit change-from-last-command term. Thus the winning Q head can change without a direct switching cost. Establishing how much that reward design caused the observed switching would require a separate controlled experiment.

This academic simulation is **not suitable for clinical deployment**. Its lumped thermal model, deterministic skin–core relationship and simplified chamber-oxygen and saturation dynamics limit physiological realism. The evaluation covers one checkpoint, five predefined scenarios and five paired seeds per scenario. Changes in patient physiology, sensor faults, actuator dynamics or combined disturbances can change the decisions and may produce unsafe actions outside the tested conditions.

The network supplies point estimates of Q without calibrated uncertainty. The DDQN receives 11 observations, including time, disturbance and scenario ID. The baseline reads core temperature and SpO₂ and maintains its own PID memory. This information asymmetry limits conclusions about the learning algorithm. Frequent switching raises an additional hardware question. The experiment measured simulated threshold avoidance. It supplies no patient-outcome evidence. SHAP establishes neither causality nor clinical safety.

The numeric definitions, example decisions, reproducibility commands and execution evidence are in `VALIDATION_REPORT.md`, `SHAP_EXPLANATION_NOTES.md`, `ENVIRONMENT_VERSIONS.md` and `reports/FINAL_TEST_RESULTS.md`.
