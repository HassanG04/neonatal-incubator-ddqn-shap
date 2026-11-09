# Decision validation

Five scenarios, five paired held-out seeds each and two controllers produced 50 episodes. All reached 200 decision steps. A reset anchor (step 0, action −1) precedes each trajectory. The 10,000 transitions record the decision state, selected action, reward from that action and next state. All 5,000 DDQN decisions carry 25 Q estimates and eleven attributions for the executed head. The baseline has no learned Q function, and the reset row contains no action to explain.

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

## Regulation by disturbance

| Scenario / controller | Core MAE (°C) | Core in band (%) | O₂ error (pp) | O₂ in band (%) | Effort | Energy (kJ) | Switches | Recovery index |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Normal / ddqn | 0.242 | 89.20 | 0.059 | 76.80 | 0.755 | 1095.93 | 187.60 | n/a |
| Normal / baseline | 0.184 | 100.00 | 0.000 | 100.00 | 0.280 | 908.82 | 114.40 | n/a |
| Ambient cooling / ddqn | 0.183 | 96.00 | 0.047 | 81.10 | 0.788 | 1282.23 | 188.40 | n/a |
| Ambient cooling / baseline | 0.212 | 96.20 | 0.000 | 100.00 | 0.356 | 1153.44 | 129.60 | n/a |
| Porthole opening / ddqn | 0.191 | 97.80 | 0.025 | 88.50 | 0.728 | 1154.25 | 189.40 | 0.8 |
| Porthole opening / baseline | 0.192 | 98.60 | 0.000 | 100.00 | 0.315 | 1021.41 | 109.60 | 0.0 |
| Initial hypothermia / ddqn | 0.192 | 96.80 | 0.026 | 87.90 | 0.702 | 1002.78 | 178.40 | 2.4 |
| Initial hypothermia / baseline | 0.187 | 98.80 | 0.000 | 100.00 | 0.290 | 933.12 | 118.80 | 2.0 |
| Oxygen desaturation / ddqn | 0.176 | 98.80 | 0.116 | 73.80 | 0.770 | 1024.65 | 178.40 | 0.4 |
| Oxygen desaturation / baseline | 0.182 | 100.00 | 0.323 | 76.30 | 0.346 | 906.39 | 128.20 | 0.0 |

Temperature error uses a 37°C setpoint and inclusive 36.5–37.5°C comfort band. Oxygen band error is zero within 91–95% and otherwise equals the distance to the nearer band edge in percentage points. Metrics use post-action states and exclude reset anchors. SD is the sample standard deviation of episode returns. Overall SD pools 25 episodes per controller. Effort sums heater fraction and normalized commanded oxygen, averaged over decisions. Heater energy integrates heater power over 45-second simulated intervals.

- **Normal.** The baseline kept both signals in band throughout all five runs. The DDQN averaged 46.4 above-band oxygen steps and 187.6 joint-action changes per episode.
- **Ambient cooling.** Ambient temperature reached 18.035°C at the last recorded state. In the second half, mean heater fractions were 0.412 for DDQN and 0.394 for baseline. DDQN had slightly lower core error. The ramp persisted through the horizon, so post-event recovery is undefined.
- **Porthole.** Heat loss increased over transition indices 50–80 inclusive. Active-window heater fractions were 0.566 and 0.516. DDQN and baseline averaged 4.4 and 2.8 core-temperature steps outside the target. From the first unaffected post-state, offsets to the first five-state in-band run averaged 0.6 and 0.0 intervals.
- **Initial hypothermia.** Reset core temperature was 35.8°C and SpO₂ was 88%. Mean heater fractions over the first 20 steps were 0.350 and 0.358. Five consecutive in-band core measurements began at mean zero-based post-state indices 2.4 and 2.0. Neither controller crossed a hard threshold.
- **Oxygen desaturation.** Mean active-window FiO₂ commands were 0.412 and 0.295. Across 1,000 post-states each, DDQN had 113 oxygen readings below band and 149 above band. Baseline had 237 below and zero above. DDQN reduced excursion severity while spending more time outside band. From the first unaffected post-state after the event, mean offsets to a five-state in-band run were 12.0 and 0.0 intervals.

The inherited recovery index searches for five consecutive in-band post-states from index 80 for porthole, 0 for hypothermia and 60 for oxygen. It can return zero at oxygen-event onset before saturation later falls. The separate post-event calculation in `reports/final_evidence/scenario_diagnostics.csv` begins at indices 81 and 121 for porthole and desaturation because forcing includes its last transition. Zero means the first eligible post-state begins a qualifying run. Each offset interval represents 45 simulated seconds. Inapplicable recovery entries are missing rather than counted as zero.

## Actuator switching

The greedy policy changed the joint action on 4,611 of 4,975 adjacent pairs (92.68%). The heater changed 3,782 times and oxygen changed 3,811 times, including 2,982 simultaneous changes. In 607 cases it returned immediately to the action from two decisions earlier. The independently replayed argmax matches every selected head. Exploration was disabled. The reward penalizes current squared actuator magnitude and has no explicit change-from-last-command term. Thus the winning Q head can change without a direct switching cost. Establishing how much that reward design caused the observed switching would require a separate controlled experiment.

This academic simulation is **not suitable for clinical deployment**. Its lumped thermal model, deterministic skin–core relationship and simplified chamber-oxygen and saturation dynamics limit physiological realism. The evaluation covers one checkpoint, five predefined scenarios and five paired seeds per scenario. Changes in patient physiology, sensor faults, actuator dynamics or combined disturbances can change the decisions and may produce unsafe actions outside the tested conditions.

The network supplies point estimates of Q without calibrated uncertainty. The DDQN receives 11 observations, including time, disturbance and scenario ID. The baseline reads core temperature and SpO₂ and maintains its own PID memory. This information asymmetry limits conclusions about the learning algorithm. Frequent switching raises an additional hardware question. The experiment measured simulated threshold avoidance. It supplies no patient-outcome evidence. SHAP establishes neither causality nor clinical safety.
