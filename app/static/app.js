// Cricket AI Assessment — Upload Page JS
// Handles: drill selection, drag-and-drop, upload, polling, state transitions

const state = {
  selectedFile: null,
  selectedDrill: 'straight_drive',
  jobId: null,
  pollTimer: null,
  stepIndex: 0,
};

const STEPS = ['step-normalize', 'step-extract', 'step-pose', 'step-metrics', 'step-scoring', 'step-report'];
const POLL_INTERVAL_MS = 3000;

// ── Drill Selection ──────────────────────────────────────
document.querySelectorAll('.drill-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.drill-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.selectedDrill = btn.dataset.value;
  });
});

// ── File Drop Zone ───────────────────────────────────────
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('videoFile');
const dropzoneContent = document.getElementById('dropzoneContent');
const dropzonePreview = document.getElementById('dropzonePreview');
const submitBtn = document.getElementById('submitBtn');

dropzone.addEventListener('dragover', e => {
  e.preventDefault();
  dropzone.classList.add('drag-over');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelect(file);
});
dropzone.addEventListener('click', e => {
  if (e.target.closest('#dropzonePreview') || e.target.id === 'removeFile') return;
  fileInput.click();
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFileSelect(fileInput.files[0]);
});
document.getElementById('removeFile').addEventListener('click', e => {
  e.stopPropagation();
  clearFile();
});

function handleFileSelect(file) {
  const ALLOWED = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!ALLOWED.includes(ext)) {
    showToast('Unsupported format. Use: ' + ALLOWED.join(', '), 'error');
    return;
  }
  state.selectedFile = file;
  document.getElementById('fileName').textContent = file.name;
  document.getElementById('fileSize').textContent = formatBytes(file.size);
  dropzoneContent.style.display = 'none';
  dropzonePreview.style.display = 'flex';
  submitBtn.disabled = false;
}

function clearFile() {
  state.selectedFile = null;
  fileInput.value = '';
  dropzoneContent.style.display = '';
  dropzonePreview.style.display = 'none';
  submitBtn.disabled = true;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ── Upload ───────────────────────────────────────────────
submitBtn.addEventListener('click', startUpload);

async function startUpload() {
  if (!state.selectedFile || !state.selectedDrill) return;

  const formData = new FormData();
  formData.append('file', state.selectedFile);
  formData.append('drill_type', state.selectedDrill);

  showProcessing();

  try {
    const res = await fetch('/upload-video', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || 'Upload failed');
      return;
    }

    state.jobId = data.job_id;
    document.getElementById('jobIdValue').textContent = state.jobId;
    document.getElementById('jobIdDisplay').style.display = 'flex';

    advanceStep(0);
    startPolling();
  } catch (err) {
    showError('Network error: ' + err.message);
  }
}

// ── Polling ──────────────────────────────────────────────
function startPolling() {
  let elapsed = 0;
  state.pollTimer = setInterval(async () => {
    elapsed += POLL_INTERVAL_MS;

    // Auto-advance steps for visual effect
    const stepIdx = Math.min(Math.floor(elapsed / 5000), STEPS.length - 1);
    advanceStep(stepIdx);

    try {
      const res = await fetch(`/status/${state.jobId}`);
      const data = await res.json();

      if (data.status === 'complete') {
        clearInterval(state.pollTimer);
        advanceStep(STEPS.length - 1);
        setTimeout(() => {
          window.location.href = `/report/${state.jobId}?format=html`;
        }, 600);
      } else if (data.status === 'failed') {
        clearInterval(state.pollTimer);
        showError(data.error || 'Processing failed');
      }
    } catch (err) {
      console.warn('Poll error:', err);
    }
  }, POLL_INTERVAL_MS);
}

// ── Step animation ───────────────────────────────────────
function advanceStep(activeIdx) {
  STEPS.forEach((stepId, i) => {
    const iconEl = document.getElementById(stepId)?.querySelector('.step-icon');
    if (!iconEl) return;
    iconEl.className = 'step-icon';
    if (i < activeIdx) {
      iconEl.className = 'step-icon step-done';
      iconEl.textContent = '✓';
      iconEl.closest('.step').style.color = '#00d4aa';
    } else if (i === activeIdx) {
      iconEl.className = 'step-icon step-active';
      iconEl.textContent = '●';
      iconEl.closest('.step').style.color = '#e8edf2';
    } else {
      iconEl.textContent = '◦';
      iconEl.closest('.step').style.color = '';
    }
  });
}

// ── UI States ────────────────────────────────────────────
function showProcessing() {
  document.getElementById('uploadCard').style.display = 'none';
  document.getElementById('processingCard').style.display = 'block';
  document.getElementById('errorCard').style.display = 'none';
}
function showError(msg) {
  document.getElementById('uploadCard').style.display = 'none';
  document.getElementById('processingCard').style.display = 'none';
  document.getElementById('errorCard').style.display = 'block';
  document.getElementById('errorMessage').textContent = msg;
}

document.getElementById('retryBtn')?.addEventListener('click', () => {
  clearInterval(state.pollTimer);
  state.jobId = null;
  state.stepIndex = 0;
  clearFile();
  document.getElementById('uploadCard').style.display = 'block';
  document.getElementById('errorCard').style.display = 'none';
  STEPS.forEach(id => {
    const el = document.getElementById(id)?.querySelector('.step-icon');
    if (el) { el.className = 'step-icon step-waiting'; el.textContent = '◦'; }
  });
});

// ── Toast ────────────────────────────────────────────────
function showToast(msg, type = 'info') {
  const toast = document.createElement('div');
  toast.style.cssText = `
    position:fixed; bottom:24px; left:50%; transform:translateX(-50%);
    background:${type === 'error' ? 'rgba(255,86,86,0.15)' : 'rgba(0,212,170,0.15)'};
    border:1px solid ${type === 'error' ? 'rgba(255,86,86,0.4)' : 'rgba(0,212,170,0.4)'};
    color:${type === 'error' ? '#ff5656' : '#00d4aa'};
    padding:12px 24px; border-radius:12px; font-size:0.875rem; font-weight:500;
    backdrop-filter:blur(12px); z-index:9999; animation:toastIn 0.3s ease;
  `;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}
