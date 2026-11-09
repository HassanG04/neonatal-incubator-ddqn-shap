# Final regression and execution evidence

The incoming unmodified repository collected 92 tests. All 92 passed with 1,255 warnings in 27.95 seconds. Three additional regression tests exercise Deep preflight failure and real Kernel fallback, propagation of an attribution error, and a seeded validation-only workflow instrumented to fail on any SHAP construction or plotting.

| Execution | Result | Wall time |
| --- | --- | ---: |
| Final suite, inherited-package project venv | 95 passed, 0 failed, 0 skipped, 1,247 warnings | 35.02 s |
| Final suite, isolated locked venv | 95 passed, 0 failed, 0 skipped, 1,247 warnings | 29.97 s |
| python -m compileall -q . in clean project copy | Exit 0 | 1.10 s |
| python run_member03.py after isolated tests and compilation | Exit 0 | 47.86 s |
| python run_member03.py in first final CLI group | Exit 0 | 64.32 s |
| python run_member03.py --skip-shap | Exit 0, no SHAP fields or SHAP output directory in fresh destination | 22.15 s |
| python run_member03.py --shap-method deep | Exit 0, actual method Deep | 64.55 s |
| python run_member03.py --shap-method kernel | Exit 0, actual method Kernel | 99.59 s |
| Independent deep audit, replay and numerical comparisons | Exit 0 | 27.77 s |
| Independent kernel audit, replay and numerical comparisons | Exit 0 | 104.50 s |
| Separate Member 02 paired evaluator, baseline and tuned seed 0 | Exit 0, 50 episode returns and switch counts match | 23.34 s |

Wall times include interpreter startup and vary with concurrent CPU load. Pytest itself reported 32.80 s in the first final suite and 28.22 s in isolation. No archived policy was optimized. Existing trainer tests exercise transient test agents.

## Warning analysis

- **1,243 gast/Python AST deprecations** arose during TensorFlow tracing. The pinned compatibility combination uses deprecated AST constructor behavior under Python 3.14.
- **3 NumPy legacy-path warnings** came from deserializing the historical checkpoint. Retaining the original bytes preserves its identity.
- **1 Gymnasium environment-checker warning** concerns checking alternative render modes without an environment registration spec.
- CLI logs also contain TensorFlow CPU/oneDNN startup and per-head tracing notices. An early plotting run reported a font-weight fallback; the final Member 03 charts use an available bold weight. The separate preserved Member 02 evaluator also emits a pandas join-keyword deprecation. The numerical replay agrees with its evaluation. No blanket warning filter was added.

## Failures and recoveries

An early added test used pytest’s per-user temporary directory and encountered Windows access denial. Its final version creates a disposable directory in the writable test project with the same assertions. The two prior error transcripts are retained. A syntax error in a development version of the independent verifier was corrected before both complete numeric reruns. An initial subprocess harness was interrupted before producing workflow evidence and replaced by direct shell execution. Its headers are retained with the exploratory transcripts. A cold dependency-resolution attempt lacked the complete set of cached packages. An offline wheelhouse then installed all 58 pinned distributions into a venv with system site packages disabled. The isolated suite and demo both passed.

## Evidence locations

`reports/final_evidence/` contains `incoming_tests.log`, `final_tests.log`, `isolated_tests.log`, `compileall.log`, `isolated_demo.log`, `default_final.log`, `deep_final.log`, `kernel_final.log` and `skip_final.log`. The numerical transcripts are `deep_numerics.log` and `kernel_numerics.log`. `review.log` records byte equality across isolated outputs and the warning counts. `environment.json` and `isolated_environment.log` identify Python and packages. `member02_evaluation.log` and `preservation.log` record the other evaluator and protected-file checks.

## Continuation verification

The final working tree was tested again: **95 passed, 1,247 warnings in 34.07 seconds**, no failures or skips, exit 0. The independent verifier exited 0 and reproduced the overall return advantage of 2.2268744786376082. A clean submission staging copy then ran `python -m compileall -q .` and `python run_member03.py`, both exit 0; all 14 canonical output files matched byte for byte. Exact timings and complete output are in `continuation_compile.log` and `continuation_demo_direct.log`. The nested-process demonstration harness stopped making progress and was interrupted; its incomplete `continuation_demo.log` is retained and is not success evidence. Direct shell execution completed. `continuation_tests.log`, `continuation_verification.log` and `continuation_preservation.log` retain the other evidence. No application source or model changes were required during this continuation.

## Checklist revision: packaged installation and fresh target environment

The supplied offline installer completed in a second, newly created environment, passed pip check and imported TensorFlow, Gymnasium, SHAP and the integrated project. The equivalent first fresh environment ran the complete suite: **95 passed, 1,247 warnings, no failures/skips, 85.01 seconds**. The official Gymnasium checker passed with its existing unregistered-render-spec warning. The integrated Deep workflow and independent verifier exited 0; 13 outputs reproduced byte for byte; the SHAP manifest differs only in its six expected output destination paths. Compilation of the clean source copy exited 0. See `reports/checklist_evidence/verification_summary.json` for timestamped command outcomes and the complete adjacent logs. Warning categories remain those documented above.

The public-index download failed, so the submission now includes the exact local wheels and an offline installer. One bootstrap attempt failed pip's Windows guard, and an installation-time compilation subprocess stalled. The corrected `python -m pip --no-compile` installation succeeded without weakening tests or suppressing warnings. The final installer checks all wheel hashes before installation. The separately supplied Python interpreter must be compatible Windows x64 CPython 3.14.
