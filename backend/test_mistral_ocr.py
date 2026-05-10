import requests
import base64
import os

def test_mistral_vision():
    image_path = "d:/OCR/Test/Image00011.jpg"
    model_name = "mistral-small3.1"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
        
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": "Transkribiere den gesamten Text in diesem Bild. Wenn es ein Dokument ist, halte dich an das Layout.",
            "images": [img_base64]
        }],
        "stream": False
    }
    
    print(f"Sending request to Ollama with {model_name}...")
    try:
        response = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        content = data.get("message", {}).get("content", "")
        print("\n--- EXTRACTED TEXT ---\n")
        print(content)
        print("\n----------------------\n")
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    test_mistral_vision()
