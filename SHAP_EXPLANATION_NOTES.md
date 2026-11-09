# Explaining the trained DDQN

The online network takes normalized observations in checkpoint order and produces 25 unbounded Q estimates. Deep SHAP applies DeepLIFT-based approximation rules to the inference-mode Keras network. Kernel SHAP fits a weighted surrogate using up to 100 coalition samples per state, with L1 selection disabled. Its prediction wrapper calls the same network. Both methods use the full 100-state reference distribution and return N × 11 × 25 for an all-head batch. Kernel sampling is reproducible for a given normalized input.

For each fixed action, the intercept is the background mean of that head’s Q. Adding eleven signed feature contributions reconstructs the selected Q. The policy-level figure aggregates each state’s greedy-head attribution. It does not measure the importance of changing a discrete action label. A contrast instead subtracts two head explanations evaluated at the same state.

A shared explainer serves validation and plotting. A Deep construction or all-head preflight failure logs the cause and selects Kernel before recording any decisions. Both manifests identify the actual method. A later attribution failure stops execution. A regression test forced the preflight failure, exercised the real Kernel fallback and verified its reference mean.

## Reference distribution

One baseline and one DDQN rollout per scenario contributed selected ticks spanning 0–198 of the 200-decision horizon. The scenario counts are 16, 20, 22, 20 and 22. Disturbance flags give 82 idle, 8 active porthole and 10 active desaturation samples. Idle can include ambient cooling, initial hypothermia and recovery. Both active disturbances and early and late states are represented. The schedule does not sample every tick or approximate a patient population. The same 100 states are used for the global figures. Correlations matter, including the modeled relationship Tsk = Tc − 0.5. Masking may form combinations that cannot occur physically.

## Global attribution in Q units

| Feature | Mean absolute φ, greedy heads | Mean absolute φ, head 22 |
| --- | --- | --- |
| phase | 0.39212 | 0.40603 |
| disturbance | 0.14808 | 0.16360 |
| scenario_id | 0.12110 | 0.10508 |
| C_O2 | 0.10396 | 0.02386 |
| Tamb | 0.10202 | 0.14065 |
| SpO2 | 0.08919 | 0.05000 |
| dTa_dt | 0.02206 | 0.02680 |
| dC_O2_dt | 0.02028 | 0.01439 |
| Tsk | 0.01733 | 0.02762 |
| Tc | 0.01546 | 0.02123 |
| Ta | 0.01240 | 0.01092 |

## Four replayed example decisions

| Scenario | Decision tick | Head | Heater / FiO₂ fractions | Background mean Q | Actual Q |
| --- | --- | --- | --- | --- | --- |
| Normal | 50 | 2 | 0.00 / 0.40 | 1.0467 | 1.1417 |
| Ambient cooling | 150 | 5 | 0.25 / 0.21 | 1.0096 | 0.1722 |
| Porthole opening | 65 | 1 | 0.00 / 0.30 | 1.0208 | 1.2279 |
| Oxygen desaturation | 90 | 9 | 0.25 / 0.60 | -0.1853 | 2.5336 |

At desaturation tick 90, heads 9 and 5 scored 2.5336 and 2.4306. The +0.1030 margin includes a −1.1949 difference in expected Q. SpO₂ contributed +0.8359, chamber oxygen +0.6512 and chamber-oxygen rate −0.3731 to this preference. Action 5 is a deliberately chosen alternative with the same heater and lower commanded oxygen. The tick specifies a near-peak forcing example, not the observed saturation minimum.

| Figure | What the values represent |
| --- | --- |
| shap_summary.png | Head 2 over all 100 sampled states. Each dot uses one signed attribution and the corresponding raw feature value. Head 2 was selected at the nominal example. |
| shap_feature_importance.png | Mean absolute contributions to fixed head 22. The command uses full heater and FiO₂ 0.40. |
| shap_policy_feature_importance.png | Mean absolute contributions to each sampled state’s greedy head, pooled over the 100 states. |
| shap_waterfall.png | Scenario 2, tick 150, seed 902000, head 5. The eleven contributions explain the difference between 1.0096 and 0.1722. |
| shap_action_contrast.png | Scenario 5, tick 90, seed 905000. Head 9 minus head 5 at identical state and heater level. |
| shap_scenario_comparison.png | One decision each from scenarios 1, 2, 3 and 5. Scenario 4 also appears in validation and the reference sample. |

The manifest’s `global_attributions.npz` retains raw states, normalized states, all Q estimates, argmax indices and the full global attribution tensor. The independent verifier reconstructs DeepLIFT rescale multipliers from the serialized dense layers. For Kernel it calls the third-party estimator directly against a separately implemented NumPy network. It checks individual features as well as intercepts and sums. These checks do not prove exact Shapley values, causation or clinical safety.

Method references: [DeepExplainer](https://shap.readthedocs.io/en/stable/generated/shap.DeepExplainer.html) and [KernelExplainer](https://shap.readthedocs.io/en/stable/generated/shap.KernelExplainer.html).
