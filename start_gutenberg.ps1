# GUTENBERG STARTUP SCRIPT v2.2
# This script launches the FastAPI backend and opens the workbench in your browser.

$ErrorActionPreference = "Stop"

Write-Host "--------------------------------------------------" -ForegroundColor Cyan
Write-Host "  🚀 GUTENBERG // AI EDITORIAL WORKBENCH v2.2" -ForegroundColor Cyan
Write-Host "--------------------------------------------------" -ForegroundColor Cyan

# 1. Check for Python
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Error: Python not found in PATH." -ForegroundColor Red
    exit
}

# 2. Cleanup Port 8000
Write-Host "Checking for existing processes on port 8000..." -ForegroundColor Gray
try {
    $portProcess = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($portProcess) {
        Write-Host "Found existing process ($($portProcess.OwningProcess)). Killing it for a clean start..." -ForegroundColor Yellow
        Stop-Process -Id $portProcess.OwningProcess -Force -ErrorAction SilentlyContinue
    }
} catch {}

# 3. Check/Start Ollama
Write-Host "Verifying Ollama Service..." -ForegroundColor Gray
try {
    $ollamaCheck = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 1
    Write-Host "✅ Ollama is online." -ForegroundColor Green
} catch {
    Write-Host "⚠️ Ollama is offline. Starting service..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Minimized
    Start-Sleep -Seconds 2
}

# 4. Check Dependencies
Write-Host "Verifying Python dependencies..." -ForegroundColor Gray
python -m pip install fastapi uvicorn requests pillow opencv-python numpy --quiet
Write-Host "✅ Dependencies verified." -ForegroundColor Green

# 5. Start Backend Service (Visible Window)
Write-Host "Launching Backend Service..." -ForegroundColor Cyan
# Using Start-Process to keep the backend logs visible in a separate window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; python backend/main.py" -WorkingDirectory $PSScriptRoot -WindowStyle Normal

# 6. Wait for service to be ready
Write-Host "Waiting for Gutenberg to initialize at http://localhost:8000..." -ForegroundColor Gray
$retries = 0
$ready = $false
while (!$ready -and $retries -lt 20) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000" -Method Head -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) { $ready = $true }
    } catch {
        # Wait and retry
    }
    if (!$ready) {
        Start-Sleep -Seconds 1
        $retries++
    }
}

if ($ready) {
    Write-Host "✅ Service is LIVE." -ForegroundColor Green
    Write-Host "Opening Workbench..." -ForegroundColor Yellow
    Start-Process "http://localhost:8000"
} else {
    Write-Host "❌ Error: Backend failed to start in time. Check the logs in the other window." -ForegroundColor Red
}

Write-Host "--------------------------------------------------" -ForegroundColor Cyan
Write-Host "✅ Deployment Process Complete. Enjoy, HAVI." -ForegroundColor Green
Write-Host "--------------------------------------------------" -ForegroundColor Cyan
