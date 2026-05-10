const API_BASE = 'http://127.0.0.1:8000/api';

// UI Elements
// UI Elements
const jobsList = document.getElementById('jobs-list');
const filesList = document.getElementById('files-list');
const dynamicParamsContainer = document.getElementById('dynamic-parameters');
const modelSelect = document.getElementById('model');

// Staging
const commitBtn = document.getElementById('commit-batch-btn');
const fileInput = document.getElementById('file-input');
const folderInput = document.getElementById('folder-input');
const stagedList = document.getElementById('staged-list');
const dropZone = document.getElementById('drop-zone');

// Preview & Canvas
const previewImg = document.getElementById('preview-img');
const imageWrapper = document.getElementById('image-wrapper');
const imageViewport = document.getElementById('image-viewport');
const polyCanvas = document.getElementById('poly-canvas');
const ctx = polyCanvas.getContext('2d');
const polyListContainer = document.getElementById('active-polygons');
const previewFilename = document.getElementById('preview-filename');

// Editors
const extractedEditor = document.getElementById('extracted-text-editor');
const translationEditor = document.getElementById('translation-editor');
const finalEditor = document.getElementById('final-editor');
const finalPageInput = document.getElementById('final-page-input');
const finalPageTotal = document.getElementById('final-page-total');
const translateBtn = document.getElementById('translate-btn');
const powerBtn = document.getElementById('power-btn');
const consoleOutput = document.getElementById('console-output');

// State
let lastLogTime = "";
let zoomScale = 1;
let translateX = 0, translateY = 0;
let isDragging = false, startX, startY;

let polygons = []; // { points: [{x,y}], order: 1 }
let currentPolyPoints = [];
let currentMouseX = null;
let currentMouseY = null;
let isDrawing = false;

let finalPages = [""];
let currentFinalPage = 0;

let stagedFiles = [];
let originalStagedFiles = [];
let activeFileIndex = -1;
let currentJobId = null;
let currentJobData = null;
let currentFileIndex = 0;
let currentRegionIndex = -1; // -1 = Combined text, 0+ = Region text

function log_event(msg, level = "INFO") {
  console.log(`[${level}] ${msg}`);
  const consoleOutput = document.getElementById('console-output');
  if (consoleOutput) {
    const line = document.createElement('div');
    line.className = 'log-line';
    const time = new Date().toLocaleTimeString();
    line.innerHTML = `<span class="log-time">[${time}]</span> <span class="log-level-${level.toLowerCase()}">${msg}</span>`;
    consoleOutput.appendChild(line);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
  }
}

function clearConsole() {
  const consoleOutput = document.getElementById('console-output');
  if (consoleOutput) consoleOutput.innerHTML = '';
}

// --- Polygon Tool Logic ---
imageViewport.addEventListener('mousedown', (e) => {
  if (e.button === 2) return; // Right click for pan
  if (!previewImg.src || previewImg.src.includes('undefined')) return;
  
  const rect = imageWrapper.getBoundingClientRect();
  const x = (e.clientX - rect.left) / (rect.width || 1);
  const y = (e.clientY - rect.top) / (rect.height || 1);

  if (x < 0 || y < 0 || x > 1 || y > 1) return;

  currentPolyPoints.push({ x, y });
  
  if (currentPolyPoints.length === 4) {
    const order = polygons.length + 1;
    polygons.push({ points: [...currentPolyPoints], order });
    currentPolyPoints = [];
    currentMouseX = null;
    currentMouseY = null;
    renderPolygons();
    updatePolyTags();
  } else {
    drawTempPoly();
  }
});

function drawTempPoly() {
  renderPolygons();
  if (currentPolyPoints.length === 0) return;
  
  const w = polyCanvas.width;
  const h = polyCanvas.height;
  ctx.strokeStyle = '#00ff00';
  ctx.fillStyle = '#00ff00';
  ctx.lineWidth = Math.max(1, w * 0.0015);
  
  ctx.beginPath();
  ctx.moveTo(currentPolyPoints[0].x * w, currentPolyPoints[0].y * h);
  currentPolyPoints.forEach(p => {
    ctx.lineTo(p.x * w, p.y * h);
  });
  ctx.stroke();

  // Draw points
  currentPolyPoints.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, ctx.lineWidth * 1.5, 0, Math.PI * 2);
    ctx.fill();
  });

  if (currentMouseX !== null && currentMouseY !== null) {
    const lastPoint = currentPolyPoints[currentPolyPoints.length - 1];
    ctx.beginPath();
    ctx.moveTo(lastPoint.x * w, lastPoint.y * h);
    ctx.lineTo(currentMouseX * w, currentMouseY * h);
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function renderPolygons() {
  if (!previewImg.naturalWidth) return;
  const w = previewImg.naturalWidth;
  const h = previewImg.naturalHeight;
  polyCanvas.width = w;
  polyCanvas.height = h;
  ctx.clearRect(0, 0, w, h);

  polygons.forEach((poly) => {
    ctx.strokeStyle = '#00ff00';
    ctx.fillStyle = 'rgba(0, 255, 0, 0.15)';
    ctx.lineWidth = Math.max(1, w * 0.0015);
    
    ctx.beginPath();
    ctx.moveTo(poly.points[0].x * w, poly.points[0].y * h);
    poly.points.slice(1).forEach(p => ctx.lineTo(p.x * w, p.y * h));
    ctx.closePath();
    ctx.stroke();
    ctx.fill();

    // Draw Points
    ctx.fillStyle = '#00ff00';
    poly.points.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, ctx.lineWidth * 1.5, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw Number
    ctx.fillStyle = '#00ff00';
    ctx.font = `bold ${w * 0.015}px IBM Plex Mono`;
    ctx.fillText(`#${poly.order}`, poly.points[0].x * w, poly.points[0].y * h - (w * 0.01));
  });
}

function updatePolyTags() {
  polyListContainer.innerHTML = '';
  polygons.sort((a, b) => a.order - b.order).forEach((poly, index) => {
    const tag = document.createElement('div');
    tag.className = 'poly-tag';
    
    const chk = document.createElement('input');
    chk.type = 'checkbox';
    chk.checked = poly.done || false;
    chk.title = "Done: Mark as completed and skip in next run";
    chk.style.marginRight = '4px';
    chk.onchange = () => {
      poly.done = chk.checked;
      saveSettings({ polygons: polygons });
    };

    tag.appendChild(chk);
    const label = document.createElement('span');
    label.textContent = `R${poly.order}`;
    tag.appendChild(label);
    
    const del = document.createElement('button');
    del.innerHTML = '×';
    del.style.cssText = "background:none;border:none;color:#ff4d4d;cursor:pointer;font-size:10px;margin-left:4px;";
    del.onclick = () => removePoly(index);
    tag.appendChild(del);

    polyListContainer.appendChild(tag);
  });
}

window.updatePolyOrder = (index, newOrder) => {
  polygons[index].order = parseInt(newOrder);
  updatePolyTags();
  renderPolygons();
};

window.removePoly = (index) => {
  polygons.splice(index, 1);
  updatePolyTags();
  renderPolygons();
};

document.getElementById('clear-poly').onclick = () => {
  polygons = [];
  currentPolyPoints = [];
  updatePolyTags();
  renderPolygons();
};

// --- Image Zoom & Pan ---
imageViewport.addEventListener('wheel', (e) => {
  e.preventDefault();
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  zoomScale = Math.min(Math.max(0.1, zoomScale * delta), 10);
  updateImageTransform();
}, { passive: false });

imageViewport.addEventListener('mousedown', (e) => {
  if (e.button !== 2) return; // Right click only for dragging
  isDragging = true;
  startX = e.clientX - translateX;
  startY = e.clientY - translateY;
  imageViewport.style.cursor = 'grabbing';
});

window.addEventListener('mousemove', (e) => {
  if (isDragging) {
    translateX = e.clientX - startX;
    translateY = e.clientY - startY;
    updateImageTransform();
  }
  
  if (currentPolyPoints.length > 0 && currentPolyPoints.length < 4) {
    // Get coordinates relative to the unscaled image dimensions
    const rect = imageWrapper.getBoundingClientRect();
    const x = (e.clientX - rect.left) / (rect.width || 1);
    const y = (e.clientY - rect.top) / (rect.height || 1);
    
    currentMouseX = x;
    currentMouseY = y;
    drawTempPoly();
  }
});

window.addEventListener('mouseup', () => {
  isDragging = false;
  imageViewport.style.cursor = 'crosshair';
});

imageViewport.oncontextmenu = (e) => e.preventDefault();

document.getElementById('zoom-in').onclick = () => { zoomScale *= 1.2; updateImageTransform(); };
document.getElementById('zoom-out').onclick = () => { zoomScale /= 1.2; updateImageTransform(); };
document.getElementById('zoom-reset').onclick = () => { zoomScale = 1; translateX = 0; translateY = 0; updateImageTransform(); };

function updateImageTransform() {
  imageWrapper.style.transform = `translate(${translateX}px, ${translateY}px) scale(${zoomScale})`;
}

function fitToViewport() {
  if (!previewImg.naturalWidth) return;
  const vWidth = imageViewport.clientWidth;
  const vHeight = imageViewport.clientHeight;
  const iWidth = previewImg.naturalWidth;
  const iHeight = previewImg.naturalHeight;
  
  const scale = Math.min(vWidth / iWidth, vHeight / iHeight, 1) * 0.95;
  zoomScale = scale;
  translateX = (vWidth - iWidth * scale) / 2;
  translateY = (vHeight - iHeight * scale) / 2;
  updateImageTransform();
}

document.getElementById('preprocess-strength').addEventListener('input', (e) => {
  document.getElementById('preprocess-strength-val').textContent = e.target.value;
});

document.getElementById('restore-btn').addEventListener('click', () => {
  if (activeFileIndex > -1) {
    stagedFiles[activeFileIndex] = originalStagedFiles[activeFileIndex];
    previewFile(activeFileIndex);
  }
});

document.getElementById('preprocess-btn').addEventListener('click', async () => {
  if (currentJobId !== null || activeFileIndex === -1 || stagedFiles.length === 0) {
    alert("Please select an image from the BATCH EXPLORER staging area to pre-process before running the batch.");
    return;
  }
  
  const btn = document.getElementById('preprocess-btn');
  const originalText = btn.textContent;
  btn.textContent = "PROCESSING...";
  btn.disabled = true;
  
  try {
    const dpi = document.getElementById('preprocess-dpi').value;
    const strength = document.getElementById('preprocess-strength').value;
    const file = stagedFiles[activeFileIndex];
    
    const formData = new FormData();
    formData.append("file", file);
    formData.append("dpi", dpi);
    formData.append("strength", strength);
    
    const res = await fetch(`${API_BASE}/preprocess`, {
      method: 'POST',
      body: formData
    });
    
    if (res.ok) {
      const blob = await res.blob();
      const newFile = new File([blob], file.name.replace(/\.[^/.]+$/, "") + "_processed.png", { type: 'image/png' });
      stagedFiles[activeFileIndex] = newFile;
      previewFile(activeFileIndex); 
    } else {
      alert("Pre-processing failed.");
    }
  } catch (e) {
    alert("Error: " + e.message);
  } finally {
    btn.textContent = originalText;
    btn.disabled = false;
  }
});

// --- Explorer / Staging Logic ---
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => handleFiles(e.target.files));
folderInput.addEventListener('change', (e) => handleFiles(e.target.files));


function handleFiles(list) {
  if (!list) return;
  for (let f of list) {
    let name = f.name;
    let count = 1;
    // Keep incrementing suffix until unique
    while (stagedFiles.find(s => s.file.name === name)) {
      const parts = f.name.split('.');
      const ext = parts.length > 1 ? parts.pop() : '';
      const base = parts.join('.');
      name = ext ? `${base}_${count}.${ext}` : `${f.name}_${count}`;
      count++;
    }
    
    // Create a new file object with the unique name
    const newFile = new File([f], name, { type: f.type });
    stagedFiles.push({ file: newFile, polygons: [] });
    originalStagedFiles.push({ file: newFile, polygons: [] });
  }
  renderStaged();
  if (activeFileIndex === -1 && stagedFiles.length > 0) previewFile(0);
}

function renderStaged() {
  stagedList.innerHTML = '';
  commitBtn.disabled = stagedFiles.length === 0;
  stagedFiles.forEach((item, i) => {
    const f = item.file;
    const div = document.createElement('div');
    div.className = `staged-item ${i === activeFileIndex ? 'active' : ''}`;

    const name = document.createElement('span');
    name.className = 'staged-name';
    name.textContent = f.name;
    name.title = f.name;
    name.onclick = () => previewFile(i);

    const removeBtn = document.createElement('button');
    removeBtn.className = 'staged-remove-btn';
    removeBtn.textContent = '×';
    removeBtn.title = 'Remove from staging';
    removeBtn.onclick = (e) => {
      e.stopPropagation();
      removeStaged(i);
    };

    div.appendChild(name);
    div.appendChild(removeBtn);
    stagedList.appendChild(div);
  });
}

function removeStaged(index) {
  // If we are removing the current file, clear polygons
  if (index === activeFileIndex) {
    polygons = [];
  }
  stagedFiles.splice(index, 1);
  originalStagedFiles.splice(index, 1);
  if (activeFileIndex >= stagedFiles.length) {
    activeFileIndex = stagedFiles.length - 1;
  }
  if (activeFileIndex >= 0) {
    previewFile(activeFileIndex);
  } else {
    activeFileIndex = -1;
    previewImg.src = '';
    previewFilename.textContent = 'NO FILE SELECTED';
    ctx.clearRect(0, 0, polyCanvas.width, polyCanvas.height);
    renderStaged();
  }
}

function previewFile(index) {
  // Save current polygons to the old file index before switching
  if (activeFileIndex !== -1 && stagedFiles[activeFileIndex]) {
    stagedFiles[activeFileIndex].polygons = [...polygons];
  }

  activeFileIndex = index;
  currentJobId = null;
  renderStaged();
  
  // Load polygons for the new file
  polygons = stagedFiles[index].polygons ? [...stagedFiles[index].polygons] : [];
  updatePolyTags();

  const restoreBtn = document.getElementById('restore-btn');
  if (stagedFiles[index] !== originalStagedFiles[index]) {
    restoreBtn.style.display = 'inline-block';
  } else {
    restoreBtn.style.display = 'none';
  }

  const f = stagedFiles[index].file;
  previewFilename.textContent = f.name;
  const reader = new FileReader();
  reader.onload = (e) => {
    // Clear canvas while loading
    ctx.clearRect(0, 0, polyCanvas.width, polyCanvas.height);
    previewImg.onload = () => {
      fitToViewport();
      renderPolygons();
    };
    previewImg.src = e.target.result;
  };
  reader.readAsDataURL(f);
}

commitBtn.onclick = async () => {
  if (currentJobId) {
    // Rerun Mode
    const res = await fetch(`${API_BASE}/jobs/${currentJobId}/rerun`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ settings: { 
        model_version: modelSelect.value, 
        prompt: document.getElementById('prompt').value,
        polygons: polygons,
        file_polygons: currentJobData.settings.file_polygons
      } })
    });
    if (res.ok) fetchJobs();
    return;
  }

  // New Batch Mode
  // Ensure the current view's polygons are saved before committing
  if (activeFileIndex !== -1 && stagedFiles[activeFileIndex]) {
    stagedFiles[activeFileIndex].polygons = [...polygons];
  }

  const formData = new FormData();
  formData.append('model_version', modelSelect.value);
  formData.append('confidence_threshold', 0.85);
  formData.append('prompt', document.getElementById('prompt').value);
  
  // Extract dynamic parameters
  const extraSettings = {};
  const dynamicInputs = document.querySelectorAll('#dynamic-parameters input');
  dynamicInputs.forEach(input => {
    const key = input.id.replace('param-', '');
    extraSettings[key] = input.type === 'range' || input.type === 'number' ? parseFloat(input.value) : input.value;
  });
  formData.append('extra_settings', JSON.stringify(extraSettings));
  
  // Create a mapping of filename to polygons
  const filePolygonsMap = {};
  stagedFiles.forEach(item => {
    filePolygonsMap[item.file.name] = item.polygons;
  });
  formData.append('file_polygons', JSON.stringify(filePolygonsMap));
  
  // Keep the global polygons field as fallback for existing backend logic if needed
  formData.append('polygons', JSON.stringify(polygons.sort((a,b) => a.order - b.order)));
  
  stagedFiles.forEach(item => formData.append('files', item.file));

  const res = await fetch(`${API_BASE}/jobs`, { method: 'POST', body: formData });
  if (res.ok) {
    const data = await res.json();
    stagedFiles = []; 
    originalStagedFiles = [];
    activeFileIndex = -1; 
    polygons = [];
    currentPolyPoints = []; 
    currentMouseX = null; currentMouseY = null;
    updatePolyTags();
    renderStaged(); 
    await fetchJobs();
    if (data.job_id) selectJob(data.job_id);
  }
};

async function fetchJobs() {
  const res = await fetch(`${API_BASE}/jobs`);
  if (res.ok) {
    const jobsList = await res.json();
    renderJobs(jobsList);
    
    // Auto-sync current job if it is processing or just finished
    if (currentJobId) {
      const activeJob = jobsList.find(j => j.id === currentJobId);
      if (activeJob && currentJobData) {
        if (activeJob.status !== 'completed' || currentJobData.status !== 'completed') {
          // Only sync if user is not actively typing in the editors
          if (document.activeElement !== extractedEditor && 
              document.activeElement !== translationEditor && 
              document.activeElement !== finalEditor) {
            selectJob(currentJobId, true);
          }
        }
      }
    }
  }
}

function renderJobs(jobs) {
  jobsList.innerHTML = '';
  jobs.forEach(j => {
    const item = document.createElement('div');
    item.className = `job-item ${j.id === currentJobId ? 'active' : ''}`;
    
    const nameDisplay = j.name || j.id.slice(-4);
    item.innerHTML = `
      <span class="job-name" title="Double click to rename">${nameDisplay}</span> 
      <div style="float:right; display:flex; gap:8px; align-items:center;">
        <span>${j.progress}%</span>
        <span class="delete-job-btn" title="Delete Batch" style="color:#ff4d4d; cursor:pointer; font-weight:bold; font-size:1.2rem; line-height:1;">×</span>
      </div>
    `;
    item.onclick = (e) => {
        if (e.target.classList.contains('job-name')) return; // handled by dblclick
        selectJob(j.id);
    };
    
    const nameEl = item.querySelector('.job-name');
    nameEl.ondblclick = (e) => {
        const newName = prompt("Enter new batch name:", nameDisplay);
        if (newName && newName !== nameDisplay) {
            renameJob(j.id, newName);
        }
    };

    const delBtn = item.querySelector('.delete-job-btn');
    delBtn.onclick = async (e) => {
        e.stopPropagation();
        if (confirm(`Are you sure you want to delete batch "${nameDisplay}"? This will PERMANENTLY delete all extracted text and images from disk.`)) {
            try {
                const res = await fetch(`${API_BASE}/jobs/${j.id}`, { method: 'DELETE' });
                if (res.ok) {
                    if (currentJobId === j.id) {
                        currentJobId = null;
                        currentJobData = null;
                        document.getElementById('file-list').innerHTML = '';
                        extractedEditor.value = '';
                        translationEditor.value = '';
                        previewImg.src = '';
                        previewFilename.textContent = 'NO JOB SELECTED';
                    }
                    await fetchJobs();
                    await fetchLogs(); // Refresh console logs immediately
                } else {
                    const err = await res.json();
                    alert("Delete failed: " + (err.detail || "Unknown error"));
                }
            } catch (err) {
                alert("Network error during delete: " + err.message);
            }
        }
    };
    
    jobsList.appendChild(item);
  });
}

async function renameJob(id, newName) {
  const res = await fetch(`${API_BASE}/jobs/${id}/rename`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: newName })
  });
  if (res.ok) {
    fetchJobs();
  }
}

async function selectJob(id, keepIndex = false) {
  currentJobId = id;
  
  const rerunBtn = document.getElementById('rerun-batch-btn');
  if (rerunBtn) rerunBtn.style.display = 'inline-flex';
  
  const addBtn = document.getElementById('add-to-batch-btn');
  if (addBtn) addBtn.style.display = 'inline-block';
  const syncBtn = document.getElementById('sync-btn');
  if (syncBtn) syncBtn.style.display = 'inline-block';

  const res = await fetch(`${API_BASE}/jobs/${id}?t=${Date.now()}`);
  if (res.ok) { 
    currentJobData = await res.json(); 
    if (currentJobData.settings) {
      if (currentJobData.settings.file_polygons && currentJobData.results.length > 0) {
        // Load polygons for the currently selected file (if keeping index) or the first file
        const targetIdx = (keepIndex && currentFileIndex < currentJobData.results.length) ? currentFileIndex : 0;
        const targetFile = currentJobData.results[targetIdx].filename;
        polygons = JSON.parse(JSON.stringify(currentJobData.settings.file_polygons[targetFile] || []));
        updatePolyTags();
      } else {
        polygons = [];
        updatePolyTags();
      }
      if (currentJobData.settings.model_version) {
        modelSelect.value = currentJobData.settings.model_version;
        updateDynamicParams(currentJobData.settings.model_version);
      }
      if (currentJobData.settings.prompt) {
        document.getElementById('prompt').value = currentJobData.settings.prompt;
      }
      if (currentJobData.settings.final_pages) {
        finalPages = currentJobData.settings.final_pages;
        currentFinalPage = 0;
        updateFinalEditorUI();
      } else {
        finalPages = [""];
        currentFinalPage = 0;
        updateFinalEditorUI();
      }
    } else {
      finalPages = [""];
      currentFinalPage = 0;
      updateFinalEditorUI();
    }
    renderFileList();
    
    // Persist selection to URL hash so it survives reloads
    window.location.hash = `job=${id}`;
    
    // Select the appropriate file
    if (keepIndex && currentFileIndex < currentJobData.results.length) {
      selectFile(currentFileIndex);
    } else {
      selectFile(0);
    }
  }
}

function updateEditors() {
  if (!currentJobData || !currentJobData.results[currentFileIndex]) return;
  const res = currentJobData.results[currentFileIndex];
  
  if (currentRegionIndex === -1) {
    extractedEditor.value = res.extracted_text || "";
    translationEditor.value = res.translation || "";
  } else {
    const order = (currentRegionIndex + 1).toString();
    const regions = res.regions || {};
    extractedEditor.value = regions[order] || "";
    // Maybe individual translations later, for now we keep file-level translation
    translationEditor.value = ""; 
  }
}

async function selectFile(idx) {
  // Save current polygons to the currentJobData before switching
  if (currentJobData && currentJobData.results[currentFileIndex]) {
    const oldFilename = currentJobData.results[currentFileIndex].filename;
    if (!currentJobData.settings.file_polygons) currentJobData.settings.file_polygons = {};
    currentJobData.settings.file_polygons[oldFilename] = JSON.parse(JSON.stringify(polygons));
  }

  currentFileIndex = idx;
  currentRegionIndex = -1;
  const res = currentJobData.results[idx];
  previewFilename.textContent = res.filename;
  
  // Load per-file polygons if available
  if (currentJobData.settings.file_polygons && currentJobData.settings.file_polygons[res.filename]) {
    polygons = JSON.parse(JSON.stringify(currentJobData.settings.file_polygons[res.filename]));
  } else {
    polygons = [];
  }
  updatePolyTags();

  // Clear canvas while loading
  ctx.clearRect(0, 0, polyCanvas.width, polyCanvas.height);
  previewImg.onload = () => {
    fitToViewport();
    renderPolygons();
  };
  previewImg.src = `${API_BASE}/jobs/${currentJobId}/files/${idx}?t=${Date.now()}`;
  
  renderFileList();
  updateEditors();
}

function selectRegion(fileIdx, polyIdx) {
  currentFileIndex = fileIdx;
  currentRegionIndex = polyIdx;
  const res = currentJobData.results[fileIdx];
  previewFilename.textContent = `${res.filename} (REGION ${polyIdx + 1})`;
  
  // Clear canvas while loading
  ctx.clearRect(0, 0, polyCanvas.width, polyCanvas.height);
  previewImg.onload = () => {
    fitToViewport();
    // Clear polygons when looking at a crop
    ctx.clearRect(0, 0, polyCanvas.width, polyCanvas.height);
  };
  previewImg.src = `${API_BASE}/jobs/${currentJobId}/files/${fileIdx}/regions/${polyIdx}?t=${Date.now()}`;
  
  const retryBtn = document.getElementById('retry-region-btn');
  if (retryBtn) retryBtn.style.display = 'inline-flex';
  
  renderFileList();
  updateEditors();
}

let resultSaveTimeouts = {};
function saveResult(idx, payload) {
  if (!currentJobId) return;
  clearTimeout(resultSaveTimeouts[idx]);
  resultSaveTimeouts[idx] = setTimeout(() => {
    fetch(`${API_BASE}/jobs/${currentJobId}/results/${idx}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  }, 500);
}

let settingsSaveTimeout = null;
function saveSettings(payload) {
  if (!currentJobId) return;
  clearTimeout(settingsSaveTimeout);
  settingsSaveTimeout = setTimeout(() => {
    fetch(`${API_BASE}/jobs/${currentJobId}/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  }, 500);
}

extractedEditor.addEventListener('input', (e) => {
  if (currentJobData && currentJobData.results[currentFileIndex]) {
    const res = currentJobData.results[currentFileIndex];
    if (currentRegionIndex === -1) {
      res.extracted_text = e.target.value;
      saveResult(currentFileIndex, { extracted_text: e.target.value });
    } else {
      const order = (currentRegionIndex + 1).toString();
      if (!res.regions) res.regions = {};
      res.regions[order] = e.target.value;
      saveResult(currentFileIndex, { regions: res.regions });
    }
  }
});

translationEditor.addEventListener('input', (e) => {
  if (currentJobData && currentJobData.results[currentFileIndex]) {
    currentJobData.results[currentFileIndex].translation = e.target.value;
    saveResult(currentFileIndex, { translation: e.target.value });
  }
});

function renderFileList() {
  filesList.innerHTML = '';
  if (!currentJobData) return;
  
  currentJobData.results.forEach((file, fileIdx) => {
    // Original File Item
    const fileItem = document.createElement('div');
    fileItem.className = `file-item ${fileIdx === currentFileIndex && currentRegionIndex === -1 ? 'active' : ''}`;
    fileItem.style.display = 'flex';
    fileItem.style.justifyContent = 'space-between';
    fileItem.style.alignItems = 'center';
    
    const fileDone = file.done ? ' <span style="color:#00ff00;">✓</span>' : '';
    
    fileItem.innerHTML = `
      <span style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">📄 ${file.filename}${fileDone}</span>
      <button onclick="removeFileFromBatch(event, ${fileIdx})" 
              style="background:none; border:none; color:#ff4d4d; cursor:pointer; font-size:14px; padding:0 4px; line-height:1;">×</button>
    `;
    fileItem.onclick = () => selectFile(fileIdx);
    filesList.appendChild(fileItem);

    // If file has independent regions, use them; otherwise use global if applicable
    const filePolygons = (currentJobData.settings.file_polygons && currentJobData.settings.file_polygons[file.filename]) 
                       ? currentJobData.settings.file_polygons[file.filename] 
                       : (currentJobData.settings.polygons || []);

    if (filePolygons.length > 0) {
      filePolygons.forEach((poly, polyIdx) => {
        const regionItem = document.createElement('div');
        regionItem.className = `file-item region-item ${fileIdx === currentFileIndex && polyIdx === currentRegionIndex ? 'active' : ''}`;
        regionItem.style.display = 'flex';
        regionItem.style.alignItems = 'center';
        regionItem.style.paddingLeft = '20px';
        
        const isDone = poly.done ? ' <span style="color:#00ff00; margin-left:auto;">✓</span>' : '';
        
        regionItem.innerHTML = `
          <span style="font-size: 11px;">└ Region ${poly.order}</span>
          ${isDone}
        `;
        regionItem.onclick = (e) => {
          e.stopPropagation();
          selectRegion(fileIdx, polyIdx);
        };
        filesList.appendChild(regionItem);
      });
    }
  });
}

window.addFilesToCurrentBatch = async (event) => {
  if (!currentJobId) return;
  const files = event.target.files;
  if (!files.length) return;
  
  const addBtn = document.getElementById('add-to-batch-btn');
  const originalText = addBtn.textContent;
  addBtn.textContent = 'UP...';
  addBtn.disabled = true;

  const formData = new FormData();
  for (let f of files) {
    formData.append('files', f);
  }
  
  try {
    log_event(`Starting upload of ${files.length} files...`, "INFO");
    const res = await fetch(`${API_BASE}/jobs/${currentJobId}/files`, {
      method: 'POST',
      body: formData
    });
    
    if (res.ok) {
      console.log("[addFilesToCurrentBatch] Upload successful. Reloading for full sync.");
      log_event(`Success: Added ${files.length} files. Refreshing...`, "SUCCESS");
      
      // Update hash and reload
      window.location.hash = `job=${currentJobId}`;
      window.location.reload();
    } else {
      const errData = await res.json();
      log_event(`Upload failed: ${errData.detail || "Server Error"}`, "ERROR");
      alert("Failed to add files: " + (errData.detail || "Server error"));
    }
  } catch (err) {
    console.error("[addFilesToCurrentBatch] Error:", err);
    log_event(`Network error: ${err.message}`, "ERROR");
    alert("Network error: " + err.message);
  } finally {
    addBtn.textContent = originalText;
    addBtn.disabled = false;
    event.target.value = '';
  }
};

window.removeFileFromBatch = async (event, index) => {
  event.stopPropagation();
  if (!confirm("Remove this image from the batch?")) return;
  
  try {
    const res = await fetch(`${API_BASE}/jobs/${currentJobId}/results/${index}`, {
      method: 'DELETE'
    });
    if (res.ok) {
      log_event(`Removed file at index ${index} from batch`, "INFO");
      const jobRes = await fetch(`${API_BASE}/jobs/${currentJobId}`);
      if (jobRes.ok) currentJobData = await jobRes.json();
      if (currentFileIndex >= currentJobData.results.length) {
        currentFileIndex = Math.max(0, currentJobData.results.length - 1);
      }
      renderFileList();
      renderResults();
    }
  } catch (err) {
    alert("Failed to remove file: " + err.message);
  }
};

window.exportBatch = () => {
  if (!currentJobData) return;
  const filename = document.getElementById('export-filename').value.trim() || 'batch_export';
  let content = "";
  currentJobData.results.forEach(res => {
    content += `--- ${res.filename} ---\n\n${res.extracted_text}\n\n`;
  });
  
  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${filename}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  log_event(`Exported batch as ${filename}.txt`, "SUCCESS");
};

// Removed duplicated selectFile, currentRegionIndex, selectRegion and renderResults functions that caused syntax errors.

document.getElementById('retry-region-btn').onclick = async () => {
  if (!currentJobId || currentFileIndex === -1 || currentRegionIndex === -1) return;
  const btn = document.getElementById('retry-region-btn');
  btn.textContent = 'RETRYING...';
  btn.disabled = true;
  
  try {
    const res = await fetch(`${API_BASE}/jobs/${currentJobId}/results/${currentFileIndex}/regions/${currentRegionIndex}/retry`, {
      method: 'POST'
    });
    const data = await res.json();
    if (data.text) {
      console.log("Retry success, adding text to editor...");
      const retryHeader = `\n\n### RETRIED REGION ${currentRegionIndex + 1}\n`;
      extractedEditor.value += retryHeader + data.text;
      saveResult(currentFileIndex, { text: extractedEditor.value });
      log_event(`Region ${currentRegionIndex + 1} retried. Result appended to EXTRACTED.MD`, "SUCCESS");
    } else if (data.error) {
      alert("Retry failed: " + data.error);
    }
  } catch (e) {
    alert("Retry error: " + e.message);
  } finally {
    btn.textContent = 'RETRY REGION ⟲';
    btn.disabled = false;
  }
};

// --- Dynamic Params ---
const MODEL_PARAMS = {
  'mistral-small3.1': [
    { id: 'temp', label: 'TEMP', type: 'range', min: 0, max: 1, step: 0.1, default: 0.1 },
    { id: 'max_tokens', label: 'MAX TOKENS', type: 'number', default: 4096 }
  ],
  'german-ocr-3.1': [
    { id: 'temp', label: 'TEMP', type: 'range', min: 0, max: 1, step: 0.1, default: 0.1 }
  ]
};

modelSelect.addEventListener('change', () => updateDynamicParams(modelSelect.value));

function updateDynamicParams(modelId) {
  const container = document.getElementById('dynamic-parameters');
  if (!container) return;
  container.innerHTML = '';
  const params = MODEL_PARAMS[modelId] || [];
  params.forEach(param => {
    const row = document.createElement('div');
    row.className = 'form-row';
    row.innerHTML = `<label>${param.label}</label>`;
    const input = document.createElement('input');
    input.type = param.type;
    input.id = `param-${param.id}`;
    input.className = param.type === 'range' ? 'minimal-slider' : 'minimal-input';
    input.value = param.default;
    if (param.type === 'range') { input.min = param.min; input.max = param.max; input.step = param.step; }
    row.appendChild(input);
    container.appendChild(row);
  });
}

// --- Global Actions ---
translateBtn.onclick = async () => {
  translateBtn.textContent = 'WORKING...';
  const res = await fetch(`${API_BASE}/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: extractedEditor.value,
      model: document.getElementById('trans-model').value,
      target_lang: document.getElementById('target-lang').value,
      tone: document.getElementById('trans-tone').value
    })
  });
  if (res.ok) {
    const data = await res.json();
    translationEditor.value = data.translation;
  }
  translateBtn.textContent = 'TRANSLATE';
};

document.getElementById('rerun-batch-btn').onclick = async () => {
  if (!currentJobId) return;
  const btn = document.getElementById('rerun-batch-btn');
  btn.textContent = 'QUEUING...';
  btn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/jobs/${currentJobId}/rerun`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_version: modelSelect.value,
        prompt: document.getElementById('prompt').value,
        polygons: polygons
      })
    });
    if (res.ok) {
      alert("Batch update queued! Check the console for progress.");
      fetchJobs();
    }
  } catch (err) {
    alert("Rerun failed: " + err.message);
  } finally {
    btn.textContent = 'RE-RUN BATCH ⟲';
    btn.disabled = false;
  }
};

document.getElementById('save-btn').onclick = async () => {
  if (!currentJobId) return;
  const btn = document.getElementById('save-btn');
  btn.textContent = 'SAVING...';
  
  // Ensure the current view's polygons are saved into the settings object
  if (currentJobData && currentJobData.results[currentFileIndex]) {
    const filename = currentJobData.results[currentFileIndex].filename;
    if (!currentJobData.settings.file_polygons) currentJobData.settings.file_polygons = {};
    currentJobData.settings.file_polygons[filename] = JSON.parse(JSON.stringify(polygons));
  }

  const res = await fetch(`${API_BASE}/jobs/${currentJobId}/settings`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(currentJobData.settings)
  });
  
  if (res.ok) {
    btn.textContent = 'SAVED ✓';
    setTimeout(() => btn.textContent = 'SAVE →', 2000);
  }
};

async function exportBatchZip() {
  if (!currentJobId) return;
  window.location.href = `${API_BASE}/jobs/${currentJobId}/export`;
}

async function importBatch(event) {
  const file = event.target.files[0];
  if (!file) return;
  
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const res = await fetch(`${API_BASE}/jobs/import`, {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (data.status === 'imported') {
      alert("Batch imported successfully!");
      fetchJobs();
    } else {
      alert("Import failed: " + data.error);
    }
  } catch (e) {
    alert("Import error: " + e.message);
  }
}

// --- Init ---
document.getElementById('theme-toggle').onclick = () => {
  document.body.classList.toggle('light-mode');
};

// Removed confidence slider logic

const statusBadge = document.getElementById('status-badge');

async function checkOllamaStatus() {
  try {
    const res = await fetch(`${API_BASE}/ollama/status?t=${Date.now()}`);
    const data = await res.json();
    if (data.status === 'ready') {
      statusBadge.textContent = 'OLLAMA SERVER READY';
      statusBadge.className = 'meta-item status-ready';
    } else {
      statusBadge.textContent = 'OLLAMA SERVER OFFLINE';
      statusBadge.className = 'meta-item status-error';
    }
  } catch (e) {
    statusBadge.textContent = 'BACKEND OFFLINE';
    statusBadge.className = 'meta-item status-error';
  }
}

setInterval(checkOllamaStatus, 5000);
checkOllamaStatus();

statusBadge.onclick = async () => {
  if (confirm('RESTART OLLAMA SERVER?')) {
    statusBadge.textContent = 'RESTARTING...';
    await fetch(`${API_BASE}/ollama/restart`, { method: 'POST' });
  }
};

dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('dragover'); };
dropZone.ondragleave = () => dropZone.classList.remove('dragover');
dropZone.ondrop = (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); handleFiles(e.dataTransfer.files); };
fileInput.onchange = (e) => handleFiles(e.target.files);
folderInput.onchange = (e) => handleFiles(e.target.files);

setInterval(fetchJobs, 3000);
updateDynamicParams(modelSelect.value);
fetchJobs().then(() => {
  // Recovery: If we have a job in the URL hash, select it automatically
  if (window.location.hash && window.location.hash.includes('job=')) {
    const jobId = window.location.hash.split('job=')[1];
    if (jobId) {
      console.log("[Recovery] Auto-selecting job from hash:", jobId);
      selectJob(jobId, true);
    }
  }
});

const REQUIRED_MODELS = [
  { id: 'mistral-small3.1:latest', label: 'Mistral Small' },
  { id: 'gemma4:31b', label: 'Gemma 4' },
  { id: 'Keyvan/german-ocr-3.1:latest', label: 'German-OCR' },
  { id: 'qwen2.5:32b', label: 'Qwen 2.5 32B' },
  { id: 'llama3.1:8b', label: 'Llama 3.1 8B' }
];

async function fetchModels() {
  try {
    const res = await fetch(`${API_BASE}/models`);
    const data = await res.json();
    const installed = data.models.map(m => m.name);

    // Update OCR Model dropdown
    const ocrSelect = document.getElementById('model');
    const currentOcrVal = ocrSelect.value;
    ocrSelect.innerHTML = '';
    
    // Update Translation Model dropdown
    const transSelect = document.getElementById('trans-model');
    const currentTransVal = transSelect.value;
    transSelect.innerHTML = '';
    
    // Populate both with all installed models
    installed.forEach(m => {
      // OCR Option
      const ocrOpt = document.createElement('option');
      ocrOpt.value = m;
      ocrOpt.textContent = m.split(':')[0].split('/').pop().toUpperCase();
      ocrSelect.appendChild(ocrOpt);

      // Translation Option (Filter for known text models to keep it clean, or just all)
      if (m.includes('qwen') || m.includes('llama') || m.includes('mistral') || m.includes('gemma')) {
        const transOpt = document.createElement('option');
        transOpt.value = m;
        transOpt.textContent = m.split(':')[0].split('/').pop().toUpperCase();
        transSelect.appendChild(transOpt);
      }
    });

    // Restore previous selections if they still exist
    if ([...ocrSelect.options].some(o => o.value === currentOcrVal)) ocrSelect.value = currentOcrVal;
    if ([...transSelect.options].some(o => o.value === currentTransVal)) transSelect.value = currentTransVal;

    checkModelIntegrity(installed);
    
    // Also update backend status while we are at it
    try {
      const statusRes = await fetch(`${API_BASE}/ollama/status`);
      const statusData = await statusRes.json();
      const isReady = statusData.status === 'ready';
      powerBtn.style.background = isReady ? '#00ff00' : '#ff4d4d';
      powerBtn.style.borderColor = isReady ? '#00ff00' : '#ff4d4d';
      powerBtn.style.color = isReady ? '#000' : '#fff';
      powerBtn.textContent = `BACKEND: ${isReady ? 'ON' : 'OFF'}`;
    } catch (e) {
      console.warn('Ollama status check failed', e);
    }
  } catch (e) {
    console.error('Error fetching models:', e);
    document.getElementById('model-status-list').innerHTML = '<div style="color:#ff4d4d">Error connecting to backend</div>';
  }
}

async function pullModel() {
  const input = document.getElementById('new-model-input');
  const btn = document.getElementById('pull-model-btn');
  const modelName = input.value.trim();
  if (!modelName) return;
  
  btn.textContent = 'PULLING...';
  btn.disabled = true;
  
  try {
    const res = await fetch(`${API_BASE}/models/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: modelName })
    });
    if (res.ok) {
      input.value = '';
      alert(`Model ${modelName} downloaded successfully!`);
      fetchModels(); // Refresh the list
    } else {
      const err = await res.json();
      alert(`Failed to pull model: ${err.detail}`);
    }
  } catch (e) {
    alert(`Error: ${e.message}`);
  } finally {
    btn.textContent = 'PULL';
    btn.disabled = false;
  }
}
document.getElementById('pull-model-btn').addEventListener('click', (e) => {
  e.preventDefault(); // Prevent form submission
  pullModel();
});

function checkModelIntegrity(installed) {
  const modelStatusList = document.getElementById('model-status-list');
  modelStatusList.innerHTML = '';
  REQUIRED_MODELS.forEach(m => {
    const isInstalled = installed.some(name => name.includes(m.id.split(':')[0]));
    const item = document.createElement('div');
    item.className = 'status-item';
    item.innerHTML = `
      <span>${m.label}</span>
      <span class="${isInstalled ? 'status-tick' : 'status-cross'}">${isInstalled ? '● READY' : '○ MISSING'}</span>
    `;
    modelStatusList.appendChild(item);
  });
}

async function fetchLogs() {
  try {
    const res = await fetch(`${API_BASE}/logs`);
    const data = await res.json();
    if (!data.logs) return;

    consoleOutput.innerHTML = '';
    data.logs.forEach(log => {
      const line = document.createElement('div');
      line.className = 'log-line';
      line.innerHTML = `
        <span class="log-time">[${log.time}]</span>
        <span class="log-level-${log.level}">${log.msg}</span>
      `;
      consoleOutput.appendChild(line);
    });
    // Auto scroll if at bottom
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
  } catch (e) {}
}

async function toggleOllama() {
  const isOff = powerBtn.textContent.includes('OFF');
  const action = isOff ? 'on' : 'off';
  
  powerBtn.disabled = true;
  powerBtn.textContent = 'WAIT...';
  
  try {
    const res = await fetch(`${API_BASE}/ollama/toggle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action })
    });
    const data = await res.json();
    
    // Status polling will update the button eventually, but let's set it now
    if (action === 'off') {
      powerBtn.style.background = '#333';
      powerBtn.style.borderColor = '#333';
      powerBtn.textContent = 'BACKEND: OFF';
    } else {
      powerBtn.style.background = '#00ff00';
      powerBtn.style.borderColor = '#00ff00';
      powerBtn.style.color = '#000';
      powerBtn.textContent = 'BACKEND: ON';
    }
  } catch (e) {
    alert("Power toggle failed: " + e.message);
  } finally {
    powerBtn.disabled = false;
  }
}

powerBtn.onclick = toggleOllama;

function clearConsole() {
  consoleOutput.innerHTML = '';
}

// Status polling
setInterval(fetchLogs, 2000);
fetchModels();
setInterval(fetchModels, 10000); 

// --- Final Edit Logic ---
function updateFinalEditorUI() {
  finalEditor.value = finalPages[currentFinalPage] || "";
  finalPageInput.value = currentFinalPage + 1;
  finalPageTotal.textContent = `/ ${finalPages.length}`;
}

finalEditor.addEventListener('input', (e) => {
  finalPages[currentFinalPage] = e.target.value;
  saveSettings({ final_pages: finalPages });
});

document.getElementById('final-add-btn').addEventListener('click', () => {
  finalPages.splice(currentFinalPage + 1, 0, "");
  currentFinalPage++;
  updateFinalEditorUI();
  saveSettings({ final_pages: finalPages });
});

document.getElementById('final-remove-btn').addEventListener('click', () => {
  if (finalPages.length > 1) {
    finalPages.splice(currentFinalPage, 1);
    if (currentFinalPage >= finalPages.length) {
      currentFinalPage = finalPages.length - 1;
    }
  } else {
    finalPages[0] = "";
  }
  updateFinalEditorUI();
  saveSettings({ final_pages: finalPages });
});

document.getElementById('final-prev-btn').addEventListener('click', () => {
  if (currentFinalPage > 0) {
    currentFinalPage--;
    updateFinalEditorUI();
  }
});

document.getElementById('final-next-btn').addEventListener('click', () => {
  if (currentFinalPage < finalPages.length - 1) {
    currentFinalPage++;
    updateFinalEditorUI();
  }
});

finalPageInput.addEventListener('change', (e) => {
  let val = parseInt(e.target.value, 10);
  if (isNaN(val) || val < 1) val = 1;
  if (val > finalPages.length) val = finalPages.length;
  currentFinalPage = val - 1;
  updateFinalEditorUI();
});

document.getElementById('final-export-txt').addEventListener('click', () => {
  const filename = document.getElementById('export-filename').value.trim() || 'page';
  finalPages.forEach((text, index) => {
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}_${index + 1}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
});

document.getElementById('final-export-pdf').addEventListener('click', () => {
  if (!window.jspdf) {
    alert("jsPDF library not loaded yet.");
    return;
  }
  const doc = new window.jspdf.jsPDF();
  const margin = 10;
  const pageHeight = doc.internal.pageSize.height;
  const pageWidth = doc.internal.pageSize.width;
  const maxLineWidth = pageWidth - margin * 2;
  
  doc.setFont("courier");
  doc.setFontSize(10);
  
  finalPages.forEach((text, index) => {
    if (index > 0) doc.addPage();
    const lines = doc.splitTextToSize(text, maxLineWidth);
    let y = margin + 10;
    
    lines.forEach(line => {
      if (y > pageHeight - margin) {
        doc.addPage();
        y = margin + 10;
      }
      doc.text(line, margin, y);
      y += 5;
    });
  });
  const filename = document.getElementById('export-filename').value.trim() || 'final_document';
  doc.save(`${filename}.pdf`);
});

document.getElementById('final-export-doc').addEventListener('click', async () => {
  if (typeof docx === 'undefined') {
    alert("docx.js library not loaded yet.");
    return;
  }

  const sections = finalPages.map(pageText => ({
    properties: {
      type: docx.SectionType.NEXT_PAGE,
    },
    children: pageText.split('\n').map(line => 
      new docx.Paragraph({
        children: [new docx.TextRun({ text: line, font: "Courier New", size: 20 })],
      })
    ),
  }));

  const doc = new docx.Document({
    sections: sections
  });

  docx.Packer.toBlob(doc).then(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const filename = document.getElementById('export-filename').value.trim() || 'final_document';
    a.download = `${filename}.docx`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
});

// Init Final UI
updateFinalEditorUI();
