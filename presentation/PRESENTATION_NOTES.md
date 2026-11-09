# Project 04 final presentation notes

One authoritative full-project deck. Member 03’s eight-slide notes remain a shorter companion. Student IDs deferred by the team.

## Slide 1: Neonatal incubator regulation

Team names supplied by Hassan. Student IDs are intentionally deferred at the user’s request. This deck describes an educational simulation, not a medical device.

Sources: Project source and frozen validation outputs

## Slide 2: Project objective and responsibilities

Member 01 owns the simulator, dataset, MDP and baseline. Member 02 owns DDQN training and evaluation. Member 03 owns explanations, decision validation and integration. All work uses the original trained checkpoint.

Sources: Project source and frozen validation outputs

## Slide 3: Synthetic data and reliability

There are no patient records. Five scenario schedules represent normal, ambient drop, porthole opening, initial hypothermia and oxygen desaturation. All claims concern one frozen checkpoint. The SHAP background reuses its 100 states for global explanations, so it is not a separate test distribution.

Sources: Project source and frozen validation outputs

## Slide 4: Related work

Kaltsogianni et al. studied 40 ventilated late-preterm/term infants, with a 92-96% target range. They found improved oxygen targeting but no significant difference in supplemental-oxygen duration. The 2025 journal issue follows first online publication in December 2024. DDQN and SHAP sources are foundational. These studies do not validate the current simulator.

Sources: Kaltsogianni et al. (2025), doi:10.1111/apa.17549; van Hasselt et al. (2016); Lundberg & Lee (2017)

## Slide 5: Markov decision process

Feature order and action mapping are verified from source and checkpoint. Scenario and disturbance indicators give the DDQN privileged information. The reward uses post-step outcomes.

Sources: src/environment.py; src/q_network.py; config.yaml

## Slide 6: Illustrative thermal and oxygen models

The simulator integrates lumped dynamics with substeps and clipping, then applies observation noise. Effective heat capacities are not measured neonatal physiology. Skin temperature is deterministically Tc minus 0.5°C. Chamber oxygen concentration is distinct from blood oxygen saturation.

Sources: src/environment.py::_physics_substep; docs/parameter_provenance.md

## Slide 7: Frozen Double DQN

The .pt file is a trusted Python pickle containing TensorFlow/Keras weights and agent metadata. It is not a PyTorch checkpoint. No training was run in this finalization. Historical configuration hash differs from the current YAML, while checkpoint inference and evaluation reproduce.

Sources: src/ddqn_agent.py; src/q_network.py; results/ddqn/ddqn_tuned_seed0/manifest.json

## Slide 8: Baseline and evaluation fairness

The baseline consumes core temperature and oxygen saturation and retains PID memory. DDQN receives all 11 features, including scenario ID, phase and disturbance flag. Matching random seeds controls environment noise but does not equalize information.

Sources: src/baseline.py; validation.py; FINAL_PROJECT_AUDIT.md

## Slide 9: SHAP methodology

Deep SHAP applies DeepLIFT-based approximation rules. Kernel SHAP uses up to 100 sampled coalitions with L1 selection disabled. A failed Deep preflight falls back to Kernel before any decisions are logged. Later attribution errors stop execution. Both paths share one explainer.

Sources: shap_explainer.py; docs/RELATED_WORK.md; SHAP_EXPLANATION_NOTES.md

## Slide 10: Background coverage and decision alignment

100 states cover ticks 0-198. Counts are 82 idle, 8 active porthole and 10 active oxygen disturbance states. Idle includes cooling and recovery, so idle is not synonymous with healthy. One hundred states are deliberately stratified, not population representative.

Sources: Project source and frozen validation outputs

## Slide 11: Feature importance depends on the Q head

Left: fixed action 22 (full heater, FiO2 0.40). Right: each state’s greedy action. Phase dominates both summaries, indicating dependence on the scripted timeline. This is model attribution, not a biological effect.

Sources: Project source and frozen validation outputs

## Slide 12: Importance across the selected actions

Selected-head importance pools each state’s greedy Q head. Phase has mean absolute attribution 0.3921. The fixed action 22 has phase attribution 0.4060. Both depend on the chosen reference distribution.

Sources: Project source and frozen validation outputs

## Slide 13: One actual ambient-cooling decision

Scenario 2, decision tick 150. The original checkpoint chooses action 5: heater fraction 0.25 and FiO2 0.21. Expected Q is about 1.0096 and actual Q is 0.1722. Features explain the estimated action value relative to the background, not whether that action is clinically appropriate.

Sources: Project source and frozen validation outputs

## Slide 14: Why action 9 outranks action 5

Scenario 5, tick 90. Both actions use heater fraction 0.25, with FiO2 0.60 versus 0.21. The Q gap is +0.1030. Positive SpO2 and chamber-oxygen contributions offset a negative expected-value difference and other contributions.

Sources: Project source and frozen validation outputs

## Slide 15: Five scenarios, different winners

Values are mean returns from five paired seeds per scenario. Baseline wins all 20 matched episodes in scenarios 1-4. DDQN wins all five oxygen-desaturation pairs. Overall weighting is equal across scenarios.

Sources: outputs/member03/validation/validation_episodes.csv

## Slide 16: Return gains come with control trade-offs

Mean episode metrics over 25 episodes per controller. Zero modeled hard-threshold terminations is a simulated observation and supplies no evidence of clinical safety.

Sources: outputs/member03/validation/validation_summary.json

## Slide 17: Actuator switching in a recorded episode

Normal scenario seed 901000, both controllers. This is one illustrative paired trajectory, not an average plot. The DDQN switches in 92.68% of adjacent decisions over all episodes. The reward penalizes actuator magnitude, not its rate of change. A causal switching-penalty experiment would require separate future training.

Sources: Project source and frozen validation outputs

## Slide 18: Verification and reproducibility

The checkpoint and Member 01/02 implementations are unchanged. Fresh installation from the public index failed, so the delivery includes the exact local wheels with provenance and checksums. The historical training YAML hash mismatch remains explicitly unresolved.

Sources: reports/checklist_evidence/; reproducibility/README.md; reports/FINAL_TEST_RESULTS.md

## Slide 19: Limitations and future work

No patient or hardware evaluation. Future changes must be evaluated separately against the preserved benchmark. SHAP cannot establish causality, uncertainty calibration or safety.

Sources: FINAL_PROJECT_HANDOVER.md; docs/RELATED_WORK.md

## Slide 20: References

All titles and links checked against primary sources. SUPPORT oxygen-targeting is distinct from the CPAP/surfactant article formerly cited. NICE motivates the 91-95% target, not the simulator’s hard termination thresholds.

Sources: Full bibliography and comparison: docs/RELATED_WORK.md

## Slide 21: Appendix: fixed-head summary

This beeswarm explains nominal action 2 across the 100 reference states. It is not a policy-label explanation. Features are normalized for network inference.

Sources: Project source and frozen validation outputs

## Slide 22: Appendix: four individual decisions

Different states and selected action heads are compared. Cross-scenario differences are descriptive and cannot establish causal effects.

Sources: Project source and frozen validation outputs
