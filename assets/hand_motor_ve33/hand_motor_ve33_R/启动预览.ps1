$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$vePython = if ($env:VE33_PYTHON) { $env:VE33_PYTHON } else { 'python' }
& $vePython (Join-Path $PSScriptRoot 'scripts/launch_preview.py') @args
exit $LASTEXITCODE
