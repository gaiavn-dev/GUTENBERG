# Gutenberg AI Editorial Workbench

**Version:** 2.2  
**Architecture:** Python / FastAPI (Backend) + Vanilla HTML5/JS (Frontend)

Gutenberg is a fully-local, privacy-first AI editorial workbench designed for batch document processing, Optical Character Recognition (OCR), visual extraction, and automated translation. It coordinates complex machine learning workloads exclusively across Ollama through a lightweight, asynchronous API and a robust brutalist UI.

## 🌟 Core Features

- **Multi-Modal AI Engine**: Orchestrates requests natively through local Ollama instances, designed for fast and secure VLM/LLM processing (`german-ocr-3.1`, `gemma4`, `mistral`, etc.). Base64-encoded crops are sent dynamically.
- **Dynamic Model Registry Pulling**: Seamlessly add new models from the Ollama registry via the UI. Massive weights are downloaded asynchronously (`asyncio.to_thread`) without freezing the workbench.
- **Intelligent Preprocessing Pipeline**: An interactive UI module running OpenCV algorithms (Gaussian thresholding, scaling, blur) to dynamically clean and prepare noisy documents before feeding them to the AI.
- **Dynamic Polygon Cropping**: Draw freehand, 4-point bounding boxes on the canvas. The engine dynamically slices the image into discrete regions for highly targeted contextual extraction.
- **Asynchronous Batch Execution**: Runs hundreds of high-VRAM inference tasks concurrently in a background `asyncio` worker thread without blocking the FastAPI event loop or the user interface.
- **Integrated AI Translator**: Built-in translation gateway applying targeted system prompts and tone modulation across multiple languages, securely wrapping local Ollama engines.
- **Power Management**: Direct API control to spin up or kill (`taskkill`) local `ollama.exe` services, instantly flushing your VRAM for gaming or other hardware-intensive workflows.
- **Zip Archival**: Full job persistence. Export complete tasks (original images, cropped regions, and metadata databases) into a standard `.zip` for offline storage or project migration.

## 🛠 Prerequisites

1. **Python 3.10+** (Ensure `pip` is in your PATH)
2. **Ollama** (Running locally on default port `11434`)
3. **PowerShell** (For Windows deployment scripts)
4. **Git** (Optional, for version control)
### Python Dependencies
The backend requires several Python packages to run optimally:
```bash
pip install fastapi uvicorn requests pillow opencv-python numpy
```

## 🚀 Quickstart & Deployment

Deploying the workbench is fully automated using the included PowerShell script. It initializes the API gateway and opens the dashboard in your default browser.

1. Clone or extract the repository to your local drive (e.g., `D:\GUTENBERG`).
2. Open PowerShell and navigate to the project directory:
   ```powershell
   cd D:\GUTENBERG
   ```
3. Run the automated startup script:
   ```powershell
   .\start_gutenberg.ps1
   ```
4. The frontend will launch at `http://localhost:8000`.

## 📁 Project Structure

```
d:\\GUTENBERG\
backend/
main.py                 # FastAPI application, queues, and model routing
frontend/
index.html              # Core brutalist UI dashboard
main.js                 # API interactions, canvas handling, and batch logic
style.css               # (If extracted) Design system tokens
jobs/                   # Automatically generated directory for batch persistence
jobs_db.json            # The synchronous thread-safe database for all tasks
start_gutenberg.ps1     # Windows deployment script
changelog.md            # Project history and version notes
README.md               # This documentation
```

## Under the Hood

### Database & Concurrency
The application relies on `jobs_db.json` within the `jobs/` directory as its single source of truth. To prevent data corruption during massive asynchronous AI operations, all writes are aggressively managed through an `asyncio.Lock()` (`jobs_lock`).

### Dynamic Parameters
Models are inherently distinct. The UI dynamically detects model capabilities and injects runtime parameters (`temp`, `max_tokens`, etc.) securely via `extra_settings` form payloads during the POST initialization.

### Memory & State Resiliency
When dragging multiple duplicate files into the workspace, the Javascript frontend safely increments filename suffixes to prevent overwriting polygon mapping coordinates. Completed files skip redundant I/O writes (like skipping `Image.crop()`) when re-running failed jobs.

## Troubleshooting

- **Server Fails to Start (Address in Use)**: If port 8000 is blocked, open PowerShell and kill the existing python instance:
  ```powershell
  $pidToKill = (Get-NetTCPConnection -LocalPort 8000).OwningProcess; Stop-Process -Id $pidToKill -Force
  ```
- **Empty Output from Ollama Models**: Ensure you have passed a valid prompt. If using a pure text model on image data, Ollama may silently drop the image. Use specific Vision tags (like `llama3.2-vision` or `Keyvan/german-ocr-3.1:latest`).
- **Backend Freezing**: Ensure you are using version `v2.1` or later; synchronous translation logic was securely wrapped in `asyncio.to_thread` to prevent thread locking.

---
*Built for scale, privacy, and brutalist efficiency.*
#
