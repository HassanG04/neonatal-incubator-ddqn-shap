# Member 03 corrections and preservation

| Area | Change | Verification |
| --- | --- | --- |
| Decision records | Added episode seed and SHAP for each executed DDQN action. Retained reset anchors and baseline null semantics. | Replayed 50 episodes and independently checked every decision vector and Q argmax. |
| Explainer selection | Integrated stages share one agent, background and explainer. Deep preflight tests all heads; fallback reason is logged. Later failures propagate. | Executed both engines, instrumented skip and forced a preflight failure in tests. |
| Kernel reference and sampling | Used all 100 states, deterministic per-input coalition sampling and no L1 feature selection. | Actual background mean and independent per-feature Kernel estimates verified. |
| Figures and provenance | Retained a global tensor and reference record schedule, eliminated repeat stochastic differences, corrected comparator labeling and chart spacing, and improved waterfall/legend placement. | Regenerated six PNGs; recomputed plotted values, compared all repeat outputs and inspected the rendered figures. |
| Statistics | Added overall sample SD and recovery-eligibility counts. A single-episode sample SD is JSON null. Added independent post-event timing to expose onset-metric ambiguity. | Independent episode, scenario and overall reductions and additional scenario diagnostics. |
| Reports and environment | Replaced stale completion claims, added a 58-package lock and synchronized duplicate reports. | Isolated suite and full workflow run; output equality and archive checks. |

Original Member 01/02 source, config, tests, models, datasets and plots are retained. No original assertion was relaxed. Three new regression tests are in `tests/test_final_workflow.py`. Original Member 03 outputs are intact within `outputs/member03_incoming.zip`. The incoming hashes and unified source diff are in `reports/final_evidence/`. No DDQN training or checkpoint write was used for these corrections.
