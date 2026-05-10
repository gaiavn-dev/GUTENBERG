import requests
import base64

def test_german_ocr():
    image_path = r"d:\OCR\jobs\job_1777288667\DSC03961.jpg"
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    payload = {
        "model": "Keyvan/german-ocr-3.1",
        "messages": [{
            "role": "user",
            "content": "Extrahiere als JSON.",
            "images": [img_base64]
        }],
        "stream": False
    }
    
    print("Submitting to Ollama (Keyvan/german-ocr-3.1)...")
    response = requests.post("http://127.0.0.1:11434/api/chat", json=payload)
    if response.ok:
        data = response.json()
        print("=== RESPONSE ===")
        print(data.get("message", {}).get("content", ""))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_german_ocr()
