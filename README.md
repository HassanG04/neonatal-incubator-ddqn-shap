# Project 04 · Neonatal incubator temperature and oxygen regulation

This academic simulator compares a rule controller with a frozen Double DQN that commands heater level and inspired oxygen fraction. Member 03 explains its action values and checks the decisions against seeded trajectories. **Not suitable for clinical deployment.**

Start with [the handover](FINAL_PROJECT_HANDOVER.md), [validation results](VALIDATION_REPORT.md), [final presentation](presentation/Project04_Final_Presentation.pptx), [slide notes](presentation/PRESENTATION_NOTES.md) and [complete test results](reports/FINAL_TEST_RESULTS.md).

## Demonstration

Run from the extracted project root on Windows x64 with CPython 3.14.6. If the Windows Python launcher is unavailable, use the full path to that interpreter for the first command.

```powershell
./reproducibility/install_offline.ps1 -Python python -Environment .venv
./.venv/Scripts/Activate.ps1
python -m pip check
python -m pytest tests/ -v
python run_member03.py
```

The supplied offline installer is the supported path. See `reproducibility/README.md` for manual commands and wheel provenance. The public-index command could not resolve the lock on this host. The interpreter itself is required separately.


Use separate destinations to retain the submitted Deep SHAP results alongside a new run.

```powershell
python shap_explainer.py --method deep --output-dir outputs/reproduction/shap
python validation.py --shap-method deep --output-dir outputs/reproduction/validation
python run_member03.py --skip-shap --output-dir outputs/validation_only
python run_member03.py --shap-method deep --output-dir outputs/deep_repeat
python run_member03.py --shap-method kernel --output-dir outputs/kernel
python tools/verify_submission.py --output-dir outputs/member03
```

The background seed defaults to 42. Evaluation seeds are fixed independently at 900000 + 1000 × scenario + episode index and pair the two controllers. The exact CLI commands without output overrides were also exercised in separate project copies. Skip mode updates validation only. Existing figures can remain from an earlier run if a directory is reused. Consult the current validation JSON for attribution status.

## Results for the submitted checkpoint

| Scenario | DDQN mean ± SD | Baseline mean ± SD | DDQN minus baseline |
| --- | --- | --- | --- |
| Normal | -25.63 ± 1.65 | -10.18 ± 0.81 | -15.44 |
| Ambient cooling | -18.47 ± 0.75 | -15.03 ± 1.75 | -3.44 |
| Porthole opening | -16.62 ± 0.98 | -11.75 ± 0.49 | -4.86 |
| Initial hypothermia | -16.87 ± 0.53 | -11.29 ± 0.68 | -5.57 |
| Oxygen desaturation | -25.71 ± 1.45 | -66.16 ± 0.89 | +40.45 |
| Overall | -20.66 ± 4.35 | -22.88 ± 22.16 | +2.23 |

## Project map

| Location | Purpose |
| --- | --- |
| src/environment.py, src/baseline.py, config.yaml | Gymnasium simulator, reward, modeled thresholds, actuator table and PID/rule controller |
| src/q_network.py, src/ddqn_agent.py, src/replay_buffer.py | Observation transform, networks, checkpoint loader and replay |
| src/train_ddqn.py, src/evaluate_controllers.py, src/rl_utils.py | Historical training, paired evaluation and shared metrics |
| shap_explainer.py, validation.py, run_member03.py | SHAP, aligned transitions and integrated CLI |
| tools/verify_submission.py | Independent network inference, environment replay, metrics and SHAP checks |
| models/, models_v1_reference/ | Frozen model collection and original reference checkpoints |
| results/, plots/, docs/ | Member 01/02 experiment records, figures and design notes |
| outputs/member03/ | Canonical Deep SHAP figures, complete attribution tensor and validation data |
| outputs/member03_incoming.zip | Original Member 03 outputs before corrections |
| reports/ | Reports, full terminal records, scenario diagnostics and per-file checksums |

The submitted `models/ddqn_tuned_seed0/best.pt` is a Python pickle of TensorFlow/Keras agent parameters and metadata. The extension is historical. The loader rebuilds the architecture and restores the arrays. Only use a trusted checkpoint. The supplied one has SHA-256 `823A2CA6988ADBF7DDB4A303DA82C1134BCFF0D18EF8C59AC21F52CC375411CE`.

The final deck contains 22 slides including two SHAP appendices. Team IDs remain deferred. The offline dependency kit and checked related work address the external review; see [checklist resolution](CHECKLIST_RESOLUTION.md).
