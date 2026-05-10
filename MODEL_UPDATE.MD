# Model Integration Guide (Ollama, Docker, Native)

This guide details how to add new OCR or LLM models to the application.

## 0. Pre-deployment: Automated Model Pull
To ensure all integrated models are available locally, run the following commands. This is required for both OCR engines and Translation workhorses.

### Current Model Stack Pull:
```powershell
# OCR Vision Models
ollama pull mistral-small3.1
ollama pull gemma4
ollama pull Keyvan/german-ocr-3.1

# Translation Models
ollama pull qwen2.5:32b
ollama pull llama3.1:8b
```

---

## 1. Adding Ollama-based Models (e.g., Mistral, Qwen)
Ollama is the easiest way to add vision-enabled models.

1. **Pull the model**: 
   ```bash
   ollama pull <model-name>
   ```
2. **Update Frontend**: Add the option to the `ENGINE` dropdown in `frontend/index.html`.
3. **Update Backend**: Add the routing in `backend/main.py` inside `process_batch_job`.
4. **Configure Parameters**: Add model-specific parameters in `frontend/main.js`'s `MODEL_PARAMS`.

## 2. Adding Docker-based Models (e.g., Nemotron)
Use this for models that require specific environments (like CUDA/CUDNN versions) not present on the host.

1. **Containerize**: Ensure you have a working `docker-compose.yml`.
2. **Path Mapping**: The app mounts `d:/OCR/jobs` to `/jobs` in the container. Ensure your script inside the container can read from `/jobs`.
3. **Integration**: Use `subprocess.run` to trigger the container as shown in `run_docker_inference`.

## 3. Adding Native Models (Transformers)
Use this for models you want to run directly via Python/Torch on the host.

1. **Install Deps**: Ensure `torch`, `transformers`, and any model-specific libs are installed.
2. **Loader Logic**: Add a loader in `load_native_model` to prevent loading multiple heavy models at once.
3. **Inference**: Implement the inference function (e.g., `run_surya_inference`).

---

## 4. Adding Dynamic Model Parameters
To expose model-specific settings (like Temperature, Top-P, or Batch Size) in the UI, update the `MODEL_PARAMS` constant in `frontend/main.js`:

```javascript
const MODEL_PARAMS = {
  'your-new-model': [
    { id: 'temp', label: 'TEMPERATURE', type: 'range', min: 0, max: 1, step: 0.1, default: 0.5 },
    { id: 'batch_size', label: 'BATCH SIZE', type: 'number', default: 1 }
  ]
};
```
These parameters are automatically bundled into an `extra_settings` JSON object and sent to the backend. In `backend/main.py`, access them via the `settings.get("extra_settings", {})` dictionary.

## 5. Adding Translation Support
The tool uses Ollama for high-quality multilingual translation. 

### Recommended Models:
- **Qwen2.5-32B**: Best for English, French, German, Vietnamese, and Chinese.
- **Mistral Small 3.1**: Excellent for European language pairs.
- **Llama 3.1-8B**: Fast, efficient baseline.

To add a new translation model:
1. Pull the model via Ollama: `ollama pull your-model-name`.
2. Add the option to the `trans-model` dropdown in `frontend/index.html`.
3. The backend `run_translation` function handles the prompt engineering automatically.

## 6. OCR-Assisted Translation Workflow
1. **Extract**: Run a batch using a vision model (e.g., Mistral or Nemotron).
2. **Review**: Check the `SOURCE.IMG` using the zoom (wheel) and pan (drag) controls.
3. **Edit**: Correct any OCR errors in the `EXTRACTED.MD` panel.
4. **Translate**: Select a target language and click `Translate →`.
5. **Refine**: Edit the translation in the `TRANSLATION.MD` panel.
6. **Export**: Use the `Export Result ↓` button to get a Markdown file with both source and translation.
