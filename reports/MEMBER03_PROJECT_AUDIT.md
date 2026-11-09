# Member 03 audit record

The independent inspection covered the observation contract, checkpoint, two attribution paths, action recording, seeded simulation, all five validation scenarios, presentation material and retention of original experiments. The original phase report has been superseded by `FINAL_PROJECT_AUDIT.md`.

| Incoming issue | Final evidence |
| --- | --- |
| Representative rows only, with no seed on trajectory rows | Each of 5,000 DDQN decisions now stores all eleven contributions and the episode seed |
| Kernel’s leading 25 background rows lacked both active disturbances | Both engines use all 100 rows; independent replay recovered 82/8/10 |
| An attribution exception could be recorded while the workflow continued | Per-decision errors propagate; preflight failure and actual Kernel fallback tested |
| Old chart labels and repeated stochastic plot calls obscured the numerical source | Manifest and tensor archive map six charts to the exact inputs and results |
| Legacy environment requirements and completion claims | Full tested lock, isolated rerun and complete new terminal evidence |

The incoming and final `config.yaml` share SHA-256 `3962d9370508b1e4b9f0fcad8c2c128c796df1539422c2381ba58b973a041578`. The historical training and tuned evaluation manifests instead record `377e80775429788b88937786a6616d9d23f2b589c8bfe950ac0f2ed7b88e1973`. A simple LF/CRLF conversion does not explain the difference. Comparison of training metadata also shows legacy YAML keys and a different representation of the learning rate. These observations do not identify the exact historical edit. The corresponding older YAML bytes were not identified in the supplied project. All twelve historical source hashes match. The current configuration reproduces the 50 overlapping historical tuned evaluation returns and switch counts. The frozen-checkpoint benchmark is reproducible; byte-for-byte reconstruction of the original training configuration remains unresolved.

The detailed audit, statistical interpretation and remaining methodological limits are in `FINAL_PROJECT_AUDIT.md`, `VALIDATION_REPORT.md` and `SHAP_EXPLANATION_NOTES.md`.
