# Kokoro FastAPI Crash Notification Script
# Called by Windows Service Control Manager on service failure
#
# NOTE: This script handles NOTIFICATION only. Service restart is handled
# automatically by Windows SCM based on the failure recovery configuration
# set during installation. Do NOT add restart logic here to avoid race conditions.

$ErrorActionPreference = "Continue"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$logFile = Join-Path $projectRoot "logs\crash-recovery.log"
$notifierPath = "C:\Services\service-crash-notifier\service-crash-notifier.exe"
$serviceName = "KokoroFastAPI"

# Log rotation settings
$maxLogSizeBytes = 5MB
$maxLogFiles = 3

function Invoke-LogRotation {
    if (Test-Path $logFile) {
        $fileInfo = Get-Item $logFile
        if ($fileInfo.Length -gt $maxLogSizeBytes) {
            # Rotate existing backup files
            for ($i = $maxLogFiles - 1; $i -ge 1; $i--) {
                $oldFile = "$logFile.$i"
                $newFile = "$logFile.$($i + 1)"
                if (Test-Path $oldFile) {
                    if ($i -eq ($maxLogFiles - 1)) {
                        Remove-Item $oldFile -Force
                    } else {
                        Move-Item $oldFile $newFile -Force
                    }
                }
            }
            # Rotate current log
            Move-Item $logFile "$logFile.1" -Force
        }
    }
}

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] [$serviceName] $Message"

    # Ensure log directory exists
    $logDir = Split-Path -Parent $logFile
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }

    # Rotate if needed
    Invoke-LogRotation

    Add-Content -Path $logFile -Value $logEntry -ErrorAction SilentlyContinue
}

Write-Log "Service crash detected" "ERROR"
Write-Log "Windows SCM will handle automatic restart based on failure recovery settings"

# Send crash notification
$timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"

if (-not (Test-Path $notifierPath)) {
    Write-Log "Crash notifier not found at: $notifierPath (notifications disabled)" "WARN"
} else {
    try {
        Write-Log "Sending crash notification..."
        & $notifierPath `
            --service $serviceName `
            --message "Kokoro FastAPI TTS service stopped unexpectedly" `
            --code 1 `
            --time $timestamp

        if ($LASTEXITCODE -eq 0) {
            Write-Log "Crash notification sent successfully"
        } else {
            Write-Log "Crash notifier exited with code $LASTEXITCODE" "WARN"
        }
    } catch {
        Write-Log "Failed to send crash notification: $_" "ERROR"
    }
}

Write-Log "Crash notification script completed"
