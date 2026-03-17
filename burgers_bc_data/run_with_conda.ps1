# Run Burgers data generation (and optional plot) using conda env "optimal-control".
# From repo root in PowerShell (with conda initialized):
#   cd burgers_bc_data
#   .\run_with_conda.ps1
# Or:  & ".\burgers_bc_data\run_with_conda.ps1"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

& conda activate optimal-control
if ($LASTEXITCODE -ne 0) {
    Write-Host "Conda env 'optimal-control' not found. Create it from repo root with:"
    Write-Host "  conda env create -f environment.yml"
    Write-Host "Then run this script again."
    exit 1
}

& python generate_burgers_bc_data.py --samples 50 --out burgers_bc_data.npz --plot
exit $LASTEXITCODE
