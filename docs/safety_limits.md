# Modeled thresholds and safety limitations

**Academic simulation only. Not suitable for clinical deployment.** The limits below are the unchanged implemented benchmark rules. They are not validated emergency limits, universal neonatal alarm settings or guarantees of patient safety.

| Variable | Reward target | Modeled termination condition |
|---|---|---|
| Core temperature Tc | Squared error around 37°C; reporting band 36.5-37.5°C | Tc < 35.5°C or Tc > 38°C |
| SpO2 | Squared distance from 91-95% | SpO2 < 85% or SpO2 > 96% |

The environment uses a -100 threshold-violation reward component. Time-limit truncation at 200 decisions is separate from termination. Boundary values themselves do not breach the strict inequalities. Validation reports episode-level threshold violations and post-state time in band. A legacy CSV field named `survived` means the episode avoided modeled termination, not patient survival.

The simulator uses effective lumped heat capacities, deterministic skin-core coupling, simplified oxygen mixing, lagged saturation and scripted disturbances. No measured infant or device data validate these dynamics. Sensor failure, unmodeled physiology, distribution shift and actuator limitations can invalidate the simulated behavior.

The greedy DDQN switches joint commands frequently and has no explicit switching penalty in its reward. Its 11 observations contain privileged scenario and disturbance information. Neither comparison with the PID/rule controller nor SHAP establishes clinical safety or causality. No model uncertainty calibration or clinical trial was performed.

## References and clinical scope

The authoritative checked bibliography is [RELATED_WORK.md](RELATED_WORK.md). NICE NG124 recommendation 1.4.2 motivates the post-stabilization 91-95% target. The SUPPORT oxygen-targeting paper is *Target Ranges of Oxygen Saturation in Extremely Preterm Infants*, DOI [10.1056/NEJMoa0911781](https://doi.org/10.1056/NEJMoa0911781). Its comparison of 85-89% and 91-95% target ranges does not establish the simulator's 85/96% abort thresholds.

Earlier documents incorrectly used the CPAP/surfactant article (DOI 10.1056/NEJMoa0911783, PMID 20472939) to support oxygen limits. That reference and the associated claims are superseded. Historical documents remain in `reports/checklist_evidence/superseded_docs.zip`. This correction changes documentation only.

