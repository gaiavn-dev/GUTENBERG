# Changelog

All notable changes to the Gutenberg AI Editorial Workbench project will be documented in this file.

## [v2.2] - 2026-05-09

### Added
- Integrated a dynamic model pull feature in the UI, allowing users to type an Ollama model tag and download it directly from the registry.
- Created an asynchronous `POST /api/models/pull` endpoint in the FastAPI backend that utilizes `asyncio.to_thread` to securely pull massive model weights without blocking the main event loop.
### Removed
- Removed Docker container orchestration (`nemotron-ocr`) from `backend/main.py` and `README.md`.
- Removed Native PyTorch processing logic (`surya-ocr`) and `HAS_TORCH` checks to pivot fully to Ollama as the exclusive inference engine.

## [v2.1] - 2026-05-08
### Added
- Real-time automatic UI synchronization in `fetchJobs` to pull job results automatically after extraction finishes, preventing silent UI failures.
- Unique file suffixing in the UI staging area to gracefully handle drag-and-dropping duplicate filenames without state overwrites.
- Support for dynamic `extra_settings` (like `temp` and `max_tokens`) correctly extracted from the frontend and appended to `process_batch_job` payload for Ollama models.

### Changed
- Refactored `run_translation` to execute concurrently via `asyncio.to_thread`, preventing the entire FastAPI event loop from being blocked by synchronous HTTP requests to Ollama.
- Updated `persist_jobs()` to use a thread-safe `asyncio.Lock()` (`jobs_lock`), eliminating JSON corruption risks caused by simultaneous thread access during batch runs.
- Enhanced backend `process_batch_job` fallback behavior to securely default to `global_polygons` even if the file-specific mapping exists but is falsely empty `[]`.
- Simplified backend model tagging. Removed brittle hardcoded tags like `Keyvan/german-ocr-3.1:latest`, redirecting raw tag definitions exclusively from the frontend select dropdowns to Ollama runners.

### Fixed
- Fixed an endpoint bug where `/api/jobs/{job_id}/files` added results silently without triggering a queue processing execution cycle.
- Fixed a silent failure where the `german-ocr-3.1` model would return an empty content string due to missing dynamic inference parameters (`num_predict`).
- Fixed I/O redundancy causing `Image.crop()` to override existing identical polygon bounds excessively during a job rerun.
