param([string]$Python = 'python', [string]$Environment = '.venv')
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & $Python -c "import sys,platform; assert sys.version_info[:2]==(3,14) and sys.platform=='win32' and platform.machine().endswith('64'), 'Requires Windows x64 CPython 3.14'"
    if ($LASTEXITCODE -ne 0) { throw 'Unsupported Python platform' }
    foreach ($line in Get-Content (Join-Path $PSScriptRoot 'WHEELS.sha256')) {
        $hash,$relative = $line -split '  ',2
        if ((Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $relative)).Hash.ToLower() -ne $hash) { throw "Wheel checksum mismatch: $relative" }
    }
    if (Test-Path -LiteralPath $Environment) { throw 'Choose a new environment directory; existing environments are preserved.' }
    & $Python -m venv --without-pip $Environment
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
    $target = Join-Path $Environment 'Scripts/python.exe'
    $priorPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = Join-Path $PSScriptRoot 'wheels/pip-26.1.2-py3-none-any.whl'
        & $target -m pip install --no-compile --no-index --find-links (Join-Path $PSScriptRoot 'wheels') pip==26.1.2 -r requirements-lock.txt
        if ($LASTEXITCODE -ne 0) { throw 'Offline installation failed' }
    } finally { $env:PYTHONPATH = $priorPath }
    & $target -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed' }
    & $target -c "import tensorflow, gymnasium, shap; import run_member03; print('Project imports passed')"
    if ($LASTEXITCODE -ne 0) { throw 'Project import check failed' }
} finally { Pop-Location }

