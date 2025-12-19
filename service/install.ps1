#Requires -RunAsAdministrator

<#
.SYNOPSIS
    Installs the Kokoro FastAPI TTS Windows service.

.DESCRIPTION
    Installs the service and configures failure recovery with email notifications.

.PARAMETER Force
    Skip confirmation prompts (for automation).

.PARAMETER WhatIf
    Preview actions without making changes.

.EXAMPLE
    .\install.ps1
    Interactive installation with prompts.

.EXAMPLE
    .\install.ps1 -Force
    Automated installation without prompts.
#>

[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Configuration
$ServiceName = "KokoroFastAPI"
$ScriptDir = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
$NotifierPath = "C:\Services\service-crash-notifier\service-crash-notifier.exe"

# Change to script directory
Set-Location $ScriptDir

# Validate crash notifier exists
if (-not (Test-Path $NotifierPath)) {
    Write-Host "WARNING: Crash notifier not found at: $NotifierPath" -ForegroundColor Yellow
    Write-Host "         Crash notifications will not work until this is installed." -ForegroundColor Yellow
    Write-Host ""
    if (-not $Force) {
        $continue = Read-Host "Continue anyway? [y/N]"
        if ($continue -ine "y") {
            exit 1
        }
    }
}

# Check if service already exists
$existingService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existingService) {
    if ($PSCmdlet.ShouldProcess($ServiceName, "Stop and uninstall existing service")) {
        Write-Host "Service already exists. Stopping and uninstalling..."
        & .\KokoroFastAPI.exe stop 2>$null
        Start-Sleep -Seconds 2
        & .\KokoroFastAPI.exe uninstall
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to uninstall existing service" -ForegroundColor Red
            exit 1
        }

        # Wait for service to be fully removed from SCM
        Write-Host "Waiting for service removal to complete..."
        $timeout = 30
        $elapsed = 0
        while ((Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) -and ($elapsed -lt $timeout)) {
            Start-Sleep -Seconds 1
            $elapsed++
        }
        if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
            Write-Host "Timeout waiting for service removal" -ForegroundColor Red
            exit 1
        }
    }
}

if ($PSCmdlet.ShouldProcess($ServiceName, "Install service")) {
    Write-Host "Installing Kokoro FastAPI TTS service..."
    & .\KokoroFastAPI.exe install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install service" -ForegroundColor Red
        exit 1
    }
}

if ($PSCmdlet.ShouldProcess($ServiceName, "Configure failure recovery")) {
    Write-Host "Configuring failure recovery..."
    $notifyScript = Join-Path $ScriptDir "notify-crash.ps1"
    # Configure SCM to: restart service (5s, 30s, 60s delays) AND run notification script
    & sc.exe failure $ServiceName reset= 3600 actions= restart/5000/restart/30000/restart/60000 command= "powershell.exe -ExecutionPolicy Bypass -File `"$notifyScript`""
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to configure failure recovery" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Service installed successfully." -ForegroundColor Green

if ($PSCmdlet.ShouldProcess($ServiceName, "Start service")) {
    Write-Host "Starting service..."
    try {
        Start-Service -Name $ServiceName -ErrorAction Stop
        Start-Sleep -Seconds 3
        $service = Get-Service -Name $ServiceName
        if ($service.Status -eq 'Running') {
            Write-Host "Service is running." -ForegroundColor Green

            # Verify health endpoint is responding
            Write-Host "Verifying health endpoint..."
            $healthUrl = "http://localhost:8880/health"
            $maxAttempts = 10
            $attempt = 0
            $healthy = $false

            while ($attempt -lt $maxAttempts -and -not $healthy) {
                try {
                    $response = Invoke-WebRequest -Uri $healthUrl -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
                    if ($response.StatusCode -eq 200) {
                        Write-Host "Health check passed - service is responding." -ForegroundColor Green
                        $healthy = $true
                    }
                } catch {
                    $attempt++
                    if ($attempt -lt $maxAttempts) {
                        Write-Host "  Waiting for service to be ready... (attempt $attempt/$maxAttempts)"
                        Start-Sleep -Seconds 3
                    }
                }
            }

            if (-not $healthy) {
                Write-Host "Warning: Service is running but health endpoint is not responding." -ForegroundColor Yellow
                Write-Host "         The service may still be initializing (loading models)." -ForegroundColor Yellow
            }
        } else {
            Write-Host "Warning: Service status is $($service.Status)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Warning: Service installed but failed to start: $_" -ForegroundColor Yellow
        Write-Host "Check logs: Get-Content `"$ScriptDir\KokoroFastAPI.out.log`"" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host ""
Write-Host "Verify failure recovery: sc.exe qfailure $ServiceName"
Write-Host "View service logs:       Get-Content `"$ScriptDir\KokoroFastAPI.out.log`""
Write-Host "View error logs:         Get-Content `"$ScriptDir\KokoroFastAPI.err.log`""
Write-Host "View recovery logs:      Get-Content `"$ProjectRoot\logs\crash-recovery.log`""
