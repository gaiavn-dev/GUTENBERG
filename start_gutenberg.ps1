# GUTENBERG STARTUP SCRIPT
# This script launches the FastAPI backend and opens the workbench in your browser.

Write-Host "--------------------------------------------------" -ForegroundColor Cyan
Write-Host "  🚀 GUTENBERG // AI EDITORIAL WORKBENCH v2.1" -ForegroundColor Cyan
Write-Host "--------------------------------------------------" -ForegroundColor Cyan

# Check for Python
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Error: Python not found in PATH." -ForegroundColor Red
    exit
}

Write-Host "Starting Backend Service..." -ForegroundColor Gray
# Launch backend in a separate process so this window doesn't hang
Start-Process python -ArgumentList "backend/main.py" -WorkingDirectory $PSScriptRoot -WindowStyle Normal

Write-Host "Waiting for service to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 3

Write-Host "Opening Workbench at http://localhost:8000..." -ForegroundColor Yellow
Start-Process "http://localhost:8000"

Write-Host "✅ Deployment Complete. Enjoy your workbench, HAVI." -ForegroundColor Green
Write-Host "--------------------------------------------------" -ForegroundColor Cyan
