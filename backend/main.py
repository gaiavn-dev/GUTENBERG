from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import time
import os
import shutil
import logging
import subprocess
import base64
import json
import requests
import threading

jobs_lock = threading.Lock()

try:
    from PIL import Image, ImageOps
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Local OCR Batch API Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)
app.mount("/jobs", StaticFiles(directory=JOBS_DIR), name="jobs")

def load_jobs():
    global jobs
    try:
        db_path = os.path.join(JOBS_DIR, "jobs_db.json")
        with open(db_path, "r", encoding="utf-8") as f:
            jobs = json.load(f)
    except FileNotFoundError:
        jobs = {}

def persist_jobs():
    try:
        db_path = os.path.join(JOBS_DIR, "jobs_db.json")
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(jobs, f, indent=2)
    except Exception as e:
        log_event(f"Failed to persist jobs: {str(e)}", "ERROR")

load_jobs()



def run_ollama_inference(image_path: str, model_name: str, system_prompt: str, settings: dict = {}):
    try:
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please process this image according to the system instructions.", "images": [img_base64]}
            ],
            "options": {
                "temperature": float(settings.get("temp", 0.1)),
                "top_p": float(settings.get("top_p", 0.9)),
                "num_predict": int(settings.get("max_tokens", 4096))
            },
            "stream": False
        }
        
        response = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=900)
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        return f"## Extracted by {model_name} (Ollama)\n\n{content}"
    except Exception as e:
        log_event(f"OLLAMA ERROR: {str(e)}", "ERROR")
        return f"[Ollama Error: {str(e)}]"

def run_translation(text: str, model: str, target_lang: str, tone: str):
    system_prompt = f"You are a professional translator. Translate the text into {target_lang} with a {tone} tone. Output ONLY the translated text. Do not add explanations or preambles."
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"<source_document>\n{text}\n</source_document>"}
            ],
            "stream": False
        }
        response = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=900)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        log_event(f"TRANSLATION ERROR: {str(e)}", "ERROR")
        return f"[Translation Error: {str(e)}]"

def process_batch_job(job_id: str, settings: dict, file_paths: list):
    total_files = len(file_paths)
    model_version = settings.get("model_version", "unknown")
    extra = settings.get("extra_settings", {})
    global_polygons = settings.get("polygons", [])
    file_polygons_map = settings.get("file_polygons", {})
    log_event(f"STARTING BATCH JOB: {job_id} ({total_files} files)", "INFO")
    
    for i, file_path in enumerate(file_paths):
        filename = os.path.basename(file_path)
        log_event(f"PROCESSING [{i+1}/{total_files}]: {filename}")
        
        # Use file-specific polygons if available, else fallback to global
        current_polygons = file_polygons_map.get(filename)
        if not current_polygons:
            current_polygons = global_polygons
        
        # Initialize or get result
        if len(jobs[job_id]["results"]) <= i:
            jobs[job_id]["results"].append({
                "filename": os.path.basename(file_path),
                "regions": {},
                "extracted_text": "",
                "translation": "",
                "done": False
            })
        
        res = jobs[job_id]["results"][i]
        if res.get("done"):
            log_event(f"SKIPPING DONE FILE: {res['filename']}")
            continue

        final_extracted_text = ""
        regions_data = res.get("regions", {})
        
        try:
            if current_polygons:
                img = Image.open(file_path)
                img = ImageOps.exif_transpose(img)
                w, h = img.size
                
                for poly_idx, poly in enumerate(current_polygons):
                    order = str(poly.get("order", poly_idx + 1))
                    
                    # Skip if region is already done
                    if poly.get("done") and order in regions_data:
                        final_extracted_text += f"### REGION {order}\n{regions_data[order]}\n\n"
                        continue

                    pts = poly["points"]
                    xs = [p["x"] * w for p in pts]
                    ys = [p["y"] * h for p in pts]
                    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
                    left, top = max(0, left), max(0, top)
                    right, bottom = min(w, right), min(h, bottom)
                    if right <= left or bottom <= top:
                        continue

                    crop_path = f"{file_path}_crop_{poly_idx}.png"
                    if not os.path.exists(crop_path):
                        crop = img.crop((left, top, right, bottom))
                        crop.save(crop_path)
                    
                    text = run_ollama_inference(crop_path, model_version, settings.get("prompt", ""), extra)
                    
                    regions_data[order] = text
                    final_extracted_text += f"### REGION {order}\n{text}\n\n"
                
                res["regions"] = regions_data
                res["extracted_text"] = final_extracted_text
            else:
                res["extracted_text"] = run_ollama_inference(file_path, model_version, settings.get("prompt", ""), extra)
                    
        except Exception as e:
            res["extracted_text"] = f"[Error processing {os.path.basename(file_path)}: {str(e)}]"
            
        jobs[job_id]["progress"] = int(((i + 1) / total_files) * 100)
        persist_jobs()
        
    jobs[job_id]["status"] = "completed"
    persist_jobs()
    log_event(f"BATCH JOB {job_id} COMPLETED", "SUCCESS")
    log_event(f"BATCH JOB {job_id} COMPLETED", "SUCCESS")

@app.post("/api/jobs/{job_id}/rerun")
async def rerun_batch_job(job_id: str, payload: dict):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    job["status"] = "queued"
    job["progress"] = 0
    
    # Use nested settings if provided, else root payload
    new_settings = payload.get("settings", payload)
    job["settings"].update(new_settings)
    persist_jobs()
    
    job_dir = os.path.join(JOBS_DIR, job_id)
    file_paths = [os.path.join(job_dir, f) for f in os.listdir(job_dir) if not f.endswith('.json') and '_crop_' not in f]
    
    await job_queue.put((job_id, job["settings"], file_paths))
    log_event(f"RE-RUNNING BATCH JOB: {job_id}", "INFO")
    return {"status": "ok"}

job_queue = asyncio.Queue()

async def worker():
    while True:
        job_data = await job_queue.get()
        job_id, settings, file_paths = job_data
        try:
            log_event(f"QUEUE: Starting job {job_id} ({len(file_paths)} files)", "INFO")
            await asyncio.to_thread(process_batch_job, job_id, settings, file_paths)
        except Exception as e:
            log_event(f"QUEUE ERROR in {job_id}: {str(e)}", "ERROR")
        finally:
            job_queue.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker())

@app.post("/api/preprocess")
async def preprocess_image(file: UploadFile = File(...), dpi: int = Form(...), strength: int = Form(15)):
    import cv2
    import numpy as np
    import io
    from fastapi.responses import StreamingResponse
    from fastapi import HTTPException
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
            
        short_edge = min(img.shape[0], img.shape[1])
        target_short_edge = int(8.27 * dpi)
        scale = target_short_edge / short_edge
        
        if scale != 1.0 and scale > 0.1:
            new_width = int(img.shape[1] * scale)
            new_height = int(img.shape[0] * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        block_size = int(target_short_edge / 40)
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(11, block_size)
        
        processed = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            block_size, 
            strength
        )
        
        success, encoded_img = cv2.imencode('.png', processed)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image")
            
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jobs")
async def create_batch_job(
    model_version: str = Form(...),
    confidence_threshold: float = Form(...),
    prompt: str = Form(...),
    polygons: str = Form("[]"),
    file_polygons: str = Form("{}"),
    extra_settings: str = Form("{}"),
    files: list[UploadFile] = File(...)
):
    job_id = f"job_{int(time.time())}"
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    saved_files = []
    for file in files:
        path = os.path.join(job_dir, file.filename)
        with open(path, "wb") as b:
            shutil.copyfileobj(file.file, b)
        saved_files.append(path)

    settings = {
        "model_version": model_version,
        "confidence_threshold": confidence_threshold,
        "prompt": prompt,
        "polygons": json.loads(polygons),
        "file_polygons": json.loads(file_polygons),
        "extra_settings": json.loads(extra_settings)
    }
    
    jobs[job_id] = {
        "id": job_id,
        "name": job_id[-4:],
        "status": "queued",
        "progress": 0,
        "settings": settings,
        "results": []
    }
    persist_jobs()
    await job_queue.put((job_id, settings, saved_files))
    return {"job_id": job_id}

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete from disk
    job_dir = os.path.join(JOBS_DIR, job_id)
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
    
    # Delete from memory
    del jobs[job_id]
    persist_jobs()
    log_event(f"JOB DELETED: {job_id}", "INFO")
    return {"status": "ok"}

@app.post("/api/translate")
async def translate_text(payload: dict):
    tone = payload.get("tone", "Professional")
    translation = await asyncio.to_thread(run_translation, payload["text"], payload["model"], payload["target_lang"], tone)
    return {"translation": translation}

system_logs = []

def log_event(msg, level="INFO"):
    entry = {"time": time.strftime("%H:%M:%S"), "msg": msg, "level": level}
    system_logs.append(entry)
    if len(system_logs) > 50:
        system_logs.pop(0)
    logging.info(f"{level}: {msg}")

@app.get("/api/logs")
async def get_logs():
    return {"logs": system_logs}

@app.get("/api/ollama/status")
async def ollama_status():
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=0.5)
        return {"status": "ready" if r.ok else "offline"}
    except Exception:
        return {"status": "offline"}

@app.post("/api/ollama/toggle")
async def toggle_ollama(payload: dict):
    action = payload.get("action") # "on" or "off"
    try:
        if action == "off":
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "ollama app.exe"], capture_output=True)
            log_event("OLLAMA SERVICE TERMINATED (GPU FLUSHED)", "WARNING")
            return {"status": "offline"}
        else:
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
            log_event("OLLAMA SERVICE STARTING...", "INFO")
            return {"status": "starting"}
    except Exception as e:
        log_event(f"TOGGLE ERROR: {str(e)}", "ERROR")
        return {"error": str(e)}

@app.get("/api/jobs/{job_id}/files/{file_index}")
async def get_job_file(job_id: str, file_index: int):
    if job_id not in jobs or file_index >= len(jobs[job_id]["results"]):
        raise HTTPException(status_code=404, detail="File not found")
    filename = jobs[job_id]["results"][file_index]["filename"]
    path = os.path.join(JOBS_DIR, job_id, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Physical file not found")
    return FileResponse(path)

@app.get("/api/jobs/{job_id}/files/{file_index}/regions/{region_index}")
async def get_job_region(job_id: str, file_index: int, region_index: int):
    if job_id not in jobs or file_index >= len(jobs[job_id]["results"]):
        raise HTTPException(status_code=404, detail="File not found")
    filename = jobs[job_id]["results"][file_index]["filename"]
    job_dir = os.path.join(JOBS_DIR, job_id)
    crop_path = os.path.join(job_dir, f"{filename}_crop_{region_index}.png")
    if not os.path.exists(crop_path):
        raise HTTPException(status_code=404, detail="Region crop not found")
    return FileResponse(crop_path)

@app.get("/api/models")
async def list_ollama_models():
    """Return the list of models currently installed in Ollama."""
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if r.ok:
            return {"models": r.json().get("models", [])}
        return {"models": []}
    except Exception as e:
        return {"models": [], "error": str(e)}

@app.post("/api/models/pull")
async def pull_model(payload: dict):
    """Instruct Ollama to pull a new model."""
    model_name = payload.get("name")
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name required")
    
    def _do_pull():
        log_event(f"Pulling new model from Ollama registry: {model_name}. This may take a few minutes...", "INFO")
        r = requests.post("http://127.0.0.1:11434/api/pull", json={"name": model_name, "stream": False}, timeout=3600)
        r.raise_for_status()
        log_event(f"Successfully pulled model: {model_name}", "INFO")

    try:
        # Run in thread so it doesn't block the API
        await asyncio.to_thread(_do_pull)
        return {"status": "success", "model": model_name}
    except Exception as e:
        log_event(f"Failed to pull model {model_name}: {str(e)}", "ERROR")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str): return jobs.get(job_id, {"error": "Not found"})

@app.get("/api/jobs")
async def list_jobs(): return sorted([{"id": k, "status": v["status"], "progress": v["progress"], "name": v.get("name", k[-4:])} for k, v in jobs.items()], key=lambda x: x["id"], reverse=True)

@app.put("/api/jobs/{job_id}/results/{file_index}")
async def update_job_result(job_id: str, file_index: int, payload: dict):
    if job_id in jobs and 0 <= file_index < len(jobs[job_id]["results"]):
        if "text" in payload:
            jobs[job_id]["results"][file_index]["extracted_text"] = payload["text"]
        if "translation" in payload:
            jobs[job_id]["results"][file_index]["translation"] = payload["translation"]
        persist_jobs()
        return {"status": "updated"}
    return {"error": "Not found"}

@app.delete("/api/jobs/{job_id}/results/{file_index}")
async def remove_job_file(job_id: str, file_index: int):
    if job_id not in jobs:
        return {"error": "Job not found"}
    job = jobs[job_id]
    if 0 <= file_index < len(job["results"]):
        res = job["results"].pop(file_index)
        # Try to delete the physical files
        job_dir = os.path.join(JOBS_DIR, job_id)
        filename = res["filename"]
        target = os.path.join(job_dir, filename)
        if os.path.exists(target):
            try:
                os.remove(target)
                # Also remove any crops
                for f in os.listdir(job_dir):
                    if f.startswith(f"{filename}_crop_"):
                        os.remove(os.path.join(job_dir, f))
            except: pass
        persist_jobs()
        return {"status": "removed"}
    return {"error": "Invalid index"}

@app.post("/api/jobs/{job_id}/files")
async def add_files_to_job(job_id: str, files: list[UploadFile] = File(...)):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    job_dir = os.path.join(JOBS_DIR, job_id)
    for f in files:
        target = os.path.join(job_dir, f.filename)
        with open(target, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        job["results"].append({
            "filename": f.filename,
            "regions": {},
            "extracted_text": "",
            "translation": "",
            "done": False
        })
    persist_jobs()
    
    file_paths = [os.path.join(job_dir, r["filename"]) for r in job["results"]]
    if job["status"] == "completed":
        job["status"] = "queued"
    await job_queue.put((job_id, job["settings"], file_paths))
    
    return {"status": "added"}

@app.put("/api/jobs/{job_id}/settings")
async def update_job_settings(job_id: str, payload: dict):
    if job_id in jobs:
        jobs[job_id]["settings"].update(payload)
        persist_jobs()
        return {"status": "updated"}
    return {"error": "Not found"}

@app.put("/api/jobs/{job_id}/rename")
async def rename_job(job_id: str, payload: dict):
    if job_id in jobs:
        jobs[job_id]["name"] = payload.get("name", job_id)
        persist_jobs()
        return {"status": "updated", "name": jobs[job_id]["name"]}
    return {"error": "Not found"}

@app.post("/api/jobs/{job_id}/results/{file_index}/regions/{region_index}/retry")
async def retry_region(job_id: str, file_index: int, region_index: int):
    job = jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
    
    settings = job["settings"]
    model_version = settings.get("model_version", "unknown")
    extra = settings.get("extra_settings", {})
    polygons = settings.get("polygons", [])
    
    if region_index >= len(polygons):
        return {"error": "Invalid region"}
    
    file_info = job["results"][file_index]
    filename = file_info["filename"]
    job_dir = os.path.join(JOBS_DIR, job_id)
    crop_path = os.path.join(job_dir, f"{filename}_crop_{region_index}.png")
    
    def _run_retry():
        return run_ollama_inference(crop_path, model_version, settings.get("prompt", ""), extra)
            
    try:
        log_event(f"RETRY REGION {region_index} on {filename}", "INFO")
        text = await asyncio.to_thread(_run_retry)
        return {"text": text}
    except Exception as e:
        log_event(f"RETRY ERROR: {str(e)}", "ERROR")
        return {"error": str(e)}

@app.get("/api/jobs/{job_id}/export")
async def export_job_zip(job_id: str):
    import zipfile
    job = jobs.get(job_id)
    if not job:
        return {"error": "Not found"}
    
    job_dir = os.path.join(JOBS_DIR, job_id)
    zip_filename = f"{job_id}_export.zip"
    zip_path = os.path.join(JOBS_DIR, zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Add all files in job dir
        for root, dirs, files in os.walk(job_dir):
            for file in files:
                zipf.write(os.path.join(root, file), file)
        
        # Add metadata.json specific to this job
        metadata = {job_id: job}
        zipf.writestr("job_data.json", json.dumps(metadata, indent=2))
        
    return FileResponse(zip_path, filename=zip_filename, background=BackgroundTask(lambda: os.remove(zip_path)))

@app.post("/api/jobs/import")
async def import_job_zip(file: UploadFile = File(...)):
    import zipfile
    temp_zip = f"temp_import_{int(time.time())}.zip"
    with open(temp_zip, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        with zipfile.ZipFile(temp_zip, 'r') as zipf:
            # Read metadata
            if "job_data.json" not in zipf.namelist():
                return {"error": "Invalid export file (missing job_data.json)"}
            
            metadata = json.loads(zipf.read("job_data.json"))
            job_id = list(metadata.keys())[0]
            
            # If job_id already exists, give it a suffix
            original_id = job_id
            counter = 1
            while job_id in jobs:
                job_id = f"{original_id}_{counter}"
                counter += 1
            
            job_dir = os.path.join(JOBS_DIR, job_id)
            os.makedirs(job_dir, exist_ok=True)
            
            # Extract everything except job_data.json
            for item in zipf.namelist():
                if item != "job_data.json":
                    zipf.extract(item, job_dir)
            
            # Update jobs database
            jobs[job_id] = metadata[original_id]
            persist_jobs()
            return {"status": "imported", "job_id": job_id}
            
    finally:
        if os.path.exists(temp_zip):
            os.remove(temp_zip)



FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
