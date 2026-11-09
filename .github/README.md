# Neonatal incubator control with Double DQN and SHAP

This academic simulation compares a learned controller with a temperature PID controller and a rule for oxygen delivery. Across five modeled scenarios, it tests how well each controller regulates temperature and oxygen, then uses SHAP to explain the learned controller’s decisions.

[Read the project report](../FINAL_PROJECT_HANDOVER.md) · [View the presentation](../presentation/Project04_Final_Presentation.pptx) · [Explore the evaluation results](../outputs/member03/validation/)

## Run the project

[Download the complete project ZIP](https://github.com/HassanG04/neonatal-incubator-ddqn-shap/releases/download/v1.0.0/neonatal-incubator-ddqn-shap-v1.0.0.zip), extract `proj_7th`, and follow the [offline setup instructions](../reproducibility/README.md). The ZIP includes all 59 dependency wheels for Windows x64. The tested Python version is 3.14.6.

The [project README](../README.md) lists the commands for running tests, reproducing the evaluation, and verifying the SHAP results. All validation uses simulated data.
