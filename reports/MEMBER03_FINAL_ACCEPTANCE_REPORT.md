# Final acceptance · Academic simulation

## Checklist revision · 8 November 2026

The revised submission now includes a rendered and validated **22-slide PPTX**, slide-by-slide notes, a checked related-work comparison, corrected oxygen references and a recorded actuator-switching plot. The old presentation plan is retired. A packaged Windows x64 / CPython 3.14 wheelhouse supplies the exact tested dependency lock through `reproducibility/install_offline.ps1`; the public index is not required by this route.

The new isolated environment passed all **95 tests** with **1,247 warnings**, no failures or skips (85.01 seconds). The Gymnasium checker passed. Fresh workflow, installation and independent-verification logs are in `reports/checklist_evidence/`. No application source, simulator configuration or checkpoint was modified in this revision.

Names supplied by the team: Ziad Kishk, Karim Ayman and Hassan Gebrill. Student IDs remain deferred at the user's request, so final administrative sign-off is pending. The historical training-configuration mismatch remains disclosed in the methods and presentation. Frozen inference/evaluation is reproducible; historical retraining is not claimed. Simulation only, not suitable for clinical deployment.

See `CHECKLIST_RESOLUTION.md` for item-by-item evidence and `presentation/Project04_Final_Presentation.pptx` for the final deck.


The frozen-model evaluation is ready for academic submission and demonstration. The provided model is not suitable for clinical deployment.

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

The default and explicit Deep runs, and the separate isolated demo, produced the same 14 files byte for byte. All four exact workflow commands exited successfully. The isolated final suite collected and passed 95 tests, with no failures or skips. SHA-256 of the original and delivered checkpoint is `823A2CA6988ADBF7DDB4A303DA82C1134BCFF0D18EF8C59AC21F52CC375411CE`.

The incoming and final `config.yaml` share SHA-256 `3962d9370508b1e4b9f0fcad8c2c128c796df1539422c2381ba58b973a041578`. The historical training and tuned evaluation manifests instead record `377e80775429788b88937786a6616d9d23f2b589c8bfe950ac0f2ed7b88e1973`. A simple LF/CRLF conversion does not explain the difference. Comparison of training metadata also shows legacy YAML keys and a different representation of the learning rate. These observations do not identify the exact historical edit. The corresponding older YAML bytes were not identified in the supplied project. All twelve historical source hashes match. The current configuration reproduces the 50 overlapping historical tuned evaluation returns and switch counts. The frozen-checkpoint benchmark is reproducible; byte-for-byte reconstruction of the original training configuration remains unresolved.

The final ZIP checksum accompanies the archive externally. `reports/SUBMISSION_FILES.sha256` enumerates its payload. The inventory itself is checked against the extracted member during archive verification.

