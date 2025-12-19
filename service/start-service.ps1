# Kokoro FastAPI Service Startup Script
# Called by WinSW to start the TTS service

$ErrorActionPreference = "Stop"

# Get project root (parent of service directory)
$projectRoot = Split-Path -Parent $PSScriptRoot

# Helper function for logging
function Write-ServiceLog {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Output "[$timestamp] [$Level] $Message"
}

Write-ServiceLog "Starting Kokoro FastAPI service..."
Write-ServiceLog "Project root: $projectRoot"

# Validate and set eSpeak library path
$espeakPaths = @(
    "C:\Program Files\eSpeak NG\libespeak-ng.dll",
    "C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll",
    "$env:ESPEAK_PATH"
)

$espeakFound = $false
foreach ($path in $espeakPaths) {
    if ($path -and (Test-Path $path)) {
        $env:PHONEMIZER_ESPEAK_LIBRARY = $path
        Write-ServiceLog "Using eSpeak library: $path"
        $espeakFound = $true
        break
    }
}

if (-not $espeakFound) {
    Write-ServiceLog "ERROR: eSpeak NG library not found. Please install eSpeak NG or set ESPEAK_PATH environment variable." "ERROR"
    exit 1
}

# Set environment variables
$env:PYTHONUTF8 = "1"
$env:PROJECT_ROOT = $projectRoot
$env:USE_GPU = "true"
$env:USE_ONNX = "false"
$env:PYTHONPATH = "$projectRoot;$projectRoot\api"
$env:MODEL_DIR = "src/models"
$env:VOICES_DIR = "src/voices/v1_0"
$env:WEB_PLAYER_PATH = "$projectRoot\web"

# Conda environment paths - configurable via environment variables
$condaRoot = if ($env:CONDA_ROOT) { $env:CONDA_ROOT } else { Join-Path $env:USERPROFILE "miniconda3" }
$condaEnv = if ($env:KOKORO_CONDA_ENV) { $env:KOKORO_CONDA_ENV } else { Join-Path $condaRoot "envs\kokoro" }

Write-ServiceLog "Conda root: $condaRoot"
Write-ServiceLog "Conda environment: $condaEnv"

# Validate conda environment exists
$pythonExe = Join-Path $condaEnv "python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-ServiceLog "ERROR: Python not found at: $pythonExe" "ERROR"
    Write-ServiceLog "Please ensure the 'kokoro' conda environment exists, or set KOKORO_CONDA_ENV to the correct path." "ERROR"
    exit 1
}

# Add conda environment to PATH
$env:PATH = "$condaEnv;$condaEnv\Scripts;$condaEnv\Library\bin;$condaRoot\condabin;$env:PATH"

# Change to project directory
Set-Location $projectRoot

# Validate main module exists
$mainModule = Join-Path $projectRoot "api\src\main.py"
if (-not (Test-Path $mainModule)) {
    Write-ServiceLog "ERROR: Main module not found at: $mainModule" "ERROR"
    exit 1
}

Write-ServiceLog "Starting uvicorn on port 8880..."

# Start uvicorn using the conda environment's python
# Using -u for unbuffered output so logs appear immediately
try {
    & $pythonExe -u -m uvicorn api.src.main:app --host 0.0.0.0 --port 8880
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-ServiceLog "Uvicorn exited with code: $exitCode" "ERROR"
        exit $exitCode
    }
} catch {
    Write-ServiceLog "Failed to start uvicorn: $_" "ERROR"
    exit 1
}
