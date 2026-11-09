# Member 03 · Eight-slide academic defense

## 1. The evaluation question

**Show** one simulated decision. Member 01 supplies the environment, reward and PID/rule baseline. Member 02 supplies the trained DDQN. Member 03 asks which action the frozen network chooses, which features contribute to its Q and what happens on the next simulation step.

**Say** “We use paired simulator episodes. One control interval represents 45 simulated seconds. There are no patient records in this experiment.”

## 2. From a state to an explanation

**Show** Q(s,a) ≈ mean background Q for a + sum of eleven signed feature contributions. Each head has its own intercept. Q is a learned estimate of future discounted return. A positive attribution can increase one head’s estimate without making it win.

**Defend** reconstruction as one check. Independent recomputation also verified per-feature values. Neither check establishes a physiological mechanism.

## 3. Explainer selection and background

**Show** 82 idle, 8 porthole and 10 desaturation states. The 100 samples come from selected early, middle and late ticks in ten baseline/DDQN trajectories. Both explainers use the same reference population.

**Explain** that Deep SHAP uses DeepLIFT-based rules, while Kernel SHAP fits weighted regressions on masked inputs. Initialization and preflight happen before the recorded decisions. A fallback is visible in the logs and manifests. Background imbalance and correlated features affect the explanation.

## 4. Which features matter across states?

**Show** `shap_summary.png`, `shap_feature_importance.png` and `shap_policy_feature_importance.png`. The beeswarm follows nominal head 2. The fixed-head bars explain command 22. The policy-level bars use whichever head is greedy at each state.

**Point out** that phase ranks first in both bar charts. Ambient temperature ranks third for head 22. Scenario ID ranks third in the greedy-head aggregate. The model has direct access to time and scenario identity.

## 5. What drove two individual Q estimates?

**Show** `shap_waterfall.png` and `shap_action_contrast.png`. At ambient cooling tick 150, action 5 scored 0.1722 against its reference mean of 1.0096. At desaturation tick 90, actions 9 and 5 use the same heater command and different oxygen commands.

**Say** “Head 9 wins by 0.1030 Q units. Oxygen-related contributions offset a negative reference difference.” Point to the −1.1949 intercept difference if asked why the plotted feature bars alone do not add to the margin.

## 6. Validation across three categories

**Show** `shap_scenario_comparison.png` alongside the scenario metrics in `VALIDATION_REPORT.md`. Normal tests stability. Cooling, porthole opening and cold start test thermal control. Desaturation stresses oxygen control. The four local examples are drawn from this five-scenario experiment.

**Explain** the row convention. Q and SHAP use the state before the action. Reward and the next state follow it. All 10,000 transitions were replayed. Distinguish recovery from event onset and recovery after the forcing ends.

## 7. DDQN versus baseline

| Scenario | DDQN mean ± SD | Baseline mean ± SD | DDQN minus baseline |
| --- | --- | --- | --- |
| Normal | -25.63 ± 1.65 | -10.18 ± 0.81 | -15.44 |
| Ambient cooling | -18.47 ± 0.75 | -15.03 ± 1.75 | -3.44 |
| Porthole opening | -16.62 ± 0.98 | -11.75 ± 0.49 | -4.86 |
| Initial hypothermia | -16.87 ± 0.53 | -11.29 ± 0.68 | -5.57 |
| Oxygen desaturation | -25.71 ± 1.45 | -66.16 ± 0.89 | +40.45 |
| Overall | -20.66 ± 4.35 | -22.88 ± 22.16 | +2.23 |

**Say** “Desaturation adds 40.45 points. Each of the other four differences favors the baseline.” The baseline also achieved higher overall target-band occupancy and used less control effort. In desaturation the DDQN reduced the severity of oxygen deviations while spending more time outside band.

## 8. Limits and future experiments

**Show** 184.44 versus 120.12 switches per 200-step episode. There are 199 opportunities to switch. No change penalty appears in the trained reward. The DDQN observes 11 features and the baseline reads two with controller memory.

**Close** with the academic scope. Candidate next experiments include an observation-matched comparison, unseen disturbance schedules, sensor/actuator uncertainty and an independently evaluated switching-cost design. This simulator and its SHAP explanations do not establish clinical safety.

## Full-project deck

These eight slides are the Member 03 companion. The authoritative full-project presentation and references are in `presentation/Project04_Final_Presentation.pptx` and `presentation/PRESENTATION_NOTES.md`.
