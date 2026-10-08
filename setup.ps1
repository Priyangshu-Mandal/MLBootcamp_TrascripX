# Windows setup: venv, dependencies, ffmpeg check, .env, tests, model/GPU check.
# Run from the project folder:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($t) { Write-Host "`n== $t ==" -ForegroundColor Cyan }
function Fail($t) { Write-Host "ERROR: $t" -ForegroundColor Red; exit 1 }
function Check($what) { if ($LASTEXITCODE -ne 0) { Fail "$what failed (exit code $LASTEXITCODE). See the message above." } }

Step "1/6 Find a regular Windows Python (python.org, 3.10 - 3.12)"
# MSYS2/MinGW/Cygwin Pythons are rejected: they build Linux-style venvs and cannot install faster-whisper.
$cands = @(, @("py", "-3.12"), @("py", "-3.11"), @("py", "-3.10"), @("py", "-3"), @("python"))
$localPy = Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue | Sort-Object FullName -Descending
foreach ($p in $localPy) { $cands += , @($p.FullName) }
$pyCmd = $null
foreach ($cand in $cands) {
    if (-not (Get-Command $cand[0] -ErrorAction SilentlyContinue)) { continue }
    $extra = @($cand | Select-Object -Skip 1)
    try {
        $info = & $cand[0] @extra -c "import sys, sysconfig, venv; print('%d.%d|%s|%s' % (sys.version_info[0], sys.version_info[1], sysconfig.get_platform(), sys.executable))" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $info) { continue }
        $ver, $plat, $exe = $info -split '\|', 3
        if ($plat -match "mingw|msys|cygwin" -or $exe -match "msys64|cygwin") { Write-Host "Skipping $($cand -join ' '): MSYS2/MinGW Python ($exe)"; continue }
        $pyCmd = $cand; Write-Host "Using $($cand -join ' ') -> Python $ver ($exe)"; break
    } catch { }
}
if (-not $pyCmd) {
    Fail "No regular Windows Python found (your 'python' is MSYS2 or the Microsoft Store placeholder). Install Python 3.12 from https://www.python.org/downloads/windows/ (tick 'Add python.exe to PATH' and keep 'py launcher'), then reopen PowerShell and rerun."
}
$pyExe = $pyCmd[0]; $pyArgs = @($pyCmd | Select-Object -Skip 1)

Step "2/6 Virtual environment + dependencies"
# Kept outside OneDrive so thousands of package files are not synced.
$venvDir = Join-Path $env:LOCALAPPDATA "meeting-assistant\venv"
$venvPy = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    & $pyExe @pyArgs -m venv $venvDir
    Check "Creating the virtual environment"
}
if (-not (Test-Path $venvPy)) { Fail "Virtual environment was not created at $venvDir." }
Write-Host "venv: $venvDir"
& $venvPy -m pip install --upgrade pip
Check "Upgrading pip"
& $venvPy -m pip install -r requirements.txt
Check "Installing requirements.txt"

Step "3/6 ffmpeg"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg not found; installing with winget."
    winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    Write-Warning "Close and reopen PowerShell, then run setup.ps1 again so ffmpeg is on PATH."
} else { Write-Host "ffmpeg found." }

Step "4/6 .env"
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env"; Write-Host "Created .env. Add GROQ_API_KEY and GEMINI_API_KEY (needed from Stage 2)." }
else { Write-Host ".env already exists." }

Step "5/6 GPU driver"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi --query-gpu=name,driver_version --format=csv }
else { Write-Warning "nvidia-smi not found: no NVIDIA driver detected. The app will use the CPU fallback." }

Step "6/6 Tests, then model/GPU check"
& $venvPy -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) { Write-Warning "Some unit tests failed (see above). Continuing with the model/GPU check." }
& $venvPy scripts\check_stt_setup.py
Write-Host "`nDone. To run the Stage 1 transcription test:" -ForegroundColor Green
Write-Host "  & `"$venvPy`" -m pipeline.stt your_recording.mp3"
