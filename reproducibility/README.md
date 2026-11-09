# Offline reproduction on Windows x64

The project supplies 59 wheel files: the exact 58 locked project distributions and pip 26.1.2. A compatible **Windows x64 CPython 3.14** interpreter is required separately. The tested interpreter is 3.14.6. No interpreter installer or operating-system runtime is bundled. Other Python/platform combinations are not verified.

From the extracted project root in PowerShell, run:

```powershell
./reproducibility/install_offline.ps1 -Python python -Environment .venv
./.venv/Scripts/python.exe -m pytest tests/ -v
./.venv/Scripts/python.exe run_member03.py --shap-method deep --output-dir outputs/reproduction
./.venv/Scripts/python.exe tools/verify_submission.py --output-dir outputs/reproduction
```

If Python is not on PATH, pass its absolute executable path to `-Python`. Choose a fresh environment path. The installer verifies every wheel SHA-256, checks the Python platform, creates an isolated environment without system packages, bootstraps pip from its supplied wheel and runs `pip check` and project imports. It does not use the public index. `--no-compile` avoids installation-time bytecode subprocesses; Python compiles imported modules normally at runtime. This does not disable project tests.

For manual installation, the equivalent operations are:

```powershell
python -m venv --without-pip .venv
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = (Resolve-Path reproducibility/wheels/pip-26.1.2-py3-none-any.whl).Path
./.venv/Scripts/python.exe -m pip install --no-compile --no-index --find-links reproducibility/wheels pip==26.1.2 -r requirements-lock.txt
$env:PYTHONPATH = $previousPythonPath
./.venv/Scripts/python.exe -m pip check
```

## Package provenance and limits

Eighteen project wheels came from the original host's pip cache. Forty were reconstructed from its installed distributions, verifying available RECORD SHA-256 entries. The pip wheel comes from that interpreter's ensurepip bundle. Reconstructed wheels are local reproduction artifacts, **not claimed to be original publisher binaries**. Their embedded license/metadata files are retained. The inventory is `WHEELS.sha256`. Original provisioning details remain in `reports/final_evidence/offline_provision.log` and `isolated_environment.log`.

The public-index attempt on 8 November 2026 failed to resolve pinned absl-py 2.5.1 on this host. This observation does not prove that package is unavailable everywhere. The portable, packaged offline procedure is the supported reproduction path for this submission. TensorFlow 2.22.0rc0 is a release candidate and Keras 3.16.0.dev2026110703 is a nightly build; neither was silently upgraded. The archive is larger because it includes all wheels.

Historical training YAML bytes still do not match the original training-manifest hash. This kit reproduces inference and evaluation of the preserved checkpoint; it does not establish byte-for-byte historical retraining. Do not retrain or overwrite the checkpoint for this submission.
