# Tested environment

## Checklist revision · 8 November 2026

The revised submission now includes a rendered and validated **22-slide PPTX**, slide-by-slide notes, a checked related-work comparison, corrected oxygen references and a recorded actuator-switching plot. The old presentation plan is retired. A packaged Windows x64 / CPython 3.14 wheelhouse supplies the exact tested dependency lock through `reproducibility/install_offline.ps1`; the public index is not required by this route.

The new isolated environment passed all **95 tests** with **1,247 warnings**, no failures or skips (85.01 seconds). The Gymnasium checker passed. Fresh workflow, installation and independent-verification logs are in `reports/checklist_evidence/`. No application source, simulator configuration or checkpoint was modified in this revision.

Names supplied by the team: Ziad Kishk, Karim Ayman and Hassan Gebrill. Student IDs remain deferred at the user's request, so final administrative sign-off is pending. The historical training-configuration mismatch remains disclosed in the methods and presentation. Frozen inference/evaluation is reproducible; historical retraining is not claimed. Simulation only, not suitable for clinical deployment.

See `CHECKLIST_RESOLUTION.md` for item-by-item evidence and `presentation/Project04_Final_Presentation.pptx` for the final deck.


Windows 11 x64, CPython 3.14.6, CPU inference. Both requirements entry points select the same 58 explicitly pinned distributions. TensorFlow is a release candidate. Keras is a nightly development build. The historical training manifest records Python 3.12.3 and TensorFlow 2.17.0.

| Component | Tested version |
| --- | --- |
| Python | 3.14.6 (64-bit) |
| tensorflow | 2.22.0rc0 |
| keras-nightly | 3.16.0.dev2026110703 |
| shap | 0.54.1 |
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| pandas | 3.1.0 |
| matplotlib | 3.11.2 |
| gymnasium | 1.4.0 |
| scikit-learn | 1.9.1 |
| pyyaml | 6.0.3 |
| pytest | 9.1.1 |
| tqdm | 4.68.2 |
| seaborn | 0.13.2 |

The first audit venv inherited installed packages. A second venv disabled system site packages. Eighteen wheels came from the host cache. Forty were reassembled for this offline isolation test from installed package files, checking available RECORD SHA-256 hashes. These local wheel artifacts are now supplied under `reproducibility/wheels/`, with provenance and SHA-256 checksums. Pip installed the exact lock in the isolated environment and reported no broken requirements. The initial index/cache installation attempt alone could not supply pinned certifi 2026.6.17. Public-index availability, other platforms and GPUs were not independently tested.

The project dependency closure excludes unrelated shared conda management packages. A base-environment consistency check had flagged conda/boltons conflicts. See `reports/final_evidence/environment.json`, `offline_provision.log` and `isolated_environment.log` for precise versions and provenance.

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
