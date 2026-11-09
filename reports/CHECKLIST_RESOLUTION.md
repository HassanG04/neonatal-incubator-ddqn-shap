# External checklist resolution

Review source: `Neonatal_Project_Final_Checklist_2026.pdf`, supplied by Hassan on 8 November 2026. This file records actual corrections, not automatic acceptance of every suggested experiment. Original restrictions against retraining, checkpoint changes and altered simulation parameters remain in force.

| Item | Resolution and evidence |
|---|---|
| 01. Actual presentation | `presentation/Project04_Final_Presentation.pptx`: 22 readable slides, native comparison tables, all six SHAP figures, an actuator plot and speaker notes. Team names included. Student IDs deferred by user, so administrative completion remains pending. |
| 02. Old presentation notes | `docs/presentation_notes.md` now points to the authoritative deck and notes. Old text is preserved only in `reports/checklist_evidence/superseded_docs.zip`. Member 03's eight-slide notes are labeled as a companion. |
| 03. Literature comparison | `docs/RELATED_WORK.md` compares a 2025 automated-oxygen trial, foundational DDQN/SHAP work and the SUPPORT oxygen-target trial. The deck includes a related-work table and checked references. |
| 04. Historical configuration | Exact old YAML bytes remain unavailable. Both methods and deck disclose the discrepancy. No original manifest, configuration or checkpoint was replaced. This satisfies the checklist's disclosure alternative, not historical retraining reproducibility. |
| 05. Portable installation | Public-index resolution failed and its log is retained. The alternative is a supplied 59-wheel kit, SHA-256 inventory and offline installer for Windows x64 / CPython 3.14. Package provenance explicitly distinguishes cached and reconstructed wheels. See `reproducibility/README.md`. |
| 06. Fresh verification | Fresh isolated environment: 95 passed, 1,247 warnings, no failures/skips. `pip check` and the Gymnasium checker passed. Workflow and independent numerical verifier rerun; complete logs in `reports/checklist_evidence/`. |
| 07. Performance trade-off | Slides 15-16 show all five scenarios, equal-scenario overall return, both time-in-band metrics and limitations. Baseline wins all 20 pairs in scenarios 1-4; DDQN wins all five desaturation pairs. |
| 08. Switching and effort | Slide 17 shows actual paired commands, with source CSV in `presentation/`. Slide 16 quantifies switches, energy and effort. Reward has no explicit switching penalty. No controller modification or retraining was performed. |
| 09. Information asymmetry | Slide 8 explicitly contrasts the baseline's Tc/SpO2 inputs with the DDQN's 11 observations, including privileged scenario, time and disturbance features. |
| 10. Physiological scope | Title and limitations slides, README and safety notes state simulation only and no clinical deployment. Exact abort thresholds are labeled educational choices. |
| 11. Medical citations | Replaced CPAP/surfactant citation as oxygen evidence with SUPPORT DOI 10.1056/NEJMoa0911781. NICE NG124 recommendation 1.4.2 supports the 91-95% target only. Removed claims that either establishes the 85/96% termination limits. |

## Preserved evidence and failed attempts

The earlier ZIP is retained locally under `_final_work/Project04_previous_final.zip`. All original source modules, checkpoints, tests and datasets remain unchanged. The test suite exercises temporary test agents but never retrains or replaces the supplied checkpoint. New files provide presentation, packaging and reproducibility support.

The first offline bootstrap invocation hit pip's Windows self-modification guard. The corrected command used `python -m pip`. An installation-time bytecode subprocess stalled; the final installer uses `--no-compile` without suppressing runtime or test warnings. Intermediate logs remain available. Presentation validation initially lacked a module-path setting and encountered unsupported filesystem hard links; the final run supplied the path and used an exclusive byte-identical copy after all validators passed. No validation checks were disabled. The deck was rendered and inspected through the authoring runtime, not opened in Microsoft PowerPoint.

## Remaining limits

Student IDs are the only deferred administrative input. Historical training-file identity remains unresolved and explicitly disclosed. Public-index installation, Linux/macOS, other Python versions, GPUs and real clinical behavior are not verified. The included offline route requires a compatible interpreter already installed. No claim of universal DDQN superiority or clinical safety is made.
