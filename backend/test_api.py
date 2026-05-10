import requests
import time
import sys

API_URL = "http://127.0.0.1:8000/api/jobs"
IMAGE_PATH = r"d:\OCR\Test\Image00011.jpg"

print(f"Submitting {IMAGE_PATH} to Surya-OCR...")
with open(IMAGE_PATH, "rb") as f:
    files = {
        "files": ("Image00011.jpg", f, "image/jpeg")
    }
    data = {
        "model_version": "surya-ocr",
        "extraction_mode": "markdown",
        "confidence_threshold": 0.85
    }
    
    response = requests.post(API_URL, files=files, data=data)

if response.status_code != 200:
    print(f"Failed to submit: {response.text}")
    sys.exit(1)

job_id = response.json()["job_id"]
print(f"Job started: {job_id}")

while True:
    status_res = requests.get(f"{API_URL}/{job_id}")
    job_data = status_res.json()
    
    status = job_data["status"]
    progress = job_data["progress"]
    
    print(f"Status: {status} | Progress: {progress}%")
    
    if status == "completed":
        print("\n=== EXTRACTED TEXT ===")
        print(job_data["results"][0]["extracted_text"])
        break
    
    if status == "failed":
        print(f"Job failed: {job_data}")
        break
        
    time.sleep(5)
