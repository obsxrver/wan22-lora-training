const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileList = document.getElementById('file-list');
const form = document.getElementById('train-form');
const startButton = document.getElementById('start-button');
const statusIndicator = document.getElementById('status-indicator');
const logOutput = document.getElementById('log-output');
const highStepEl = document.getElementById('high-step');
const highLossEl = document.getElementById('high-loss');
const lowStepEl = document.getElementById('low-step');
const lowLossEl = document.getElementById('low-loss');

let files = [];
let logLines = [];

const chart = new Chart(document.getElementById('loss-chart'), {
  type: 'line',
  data: {
    datasets: [
      {
        label: 'High noise',
        borderColor: '#1f77b4',
        backgroundColor: 'rgba(31, 119, 180, 0.15)',
        data: [],
        tension: 0.3,
      },
      {
        label: 'Low noise',
        borderColor: '#ff7f0e',
        backgroundColor: 'rgba(255, 127, 14, 0.15)',
        data: [],
        tension: 0.3,
      },
    ],
  },
  options: {
    responsive: true,
    animation: false,
    scales: {
      x: {
        title: { display: true, text: 'Step' },
      },
      y: {
        title: { display: true, text: 'Average loss' },
      },
    },
    plugins: {
      legend: { display: true },
    },
  },
});

function renderFileList() {
  fileList.innerHTML = '';
  if (!files.length) {
    fileList.innerHTML = '<li class="muted">No files added yet.</li>';
    return;
  }

  files.forEach((file, index) => {
    const li = document.createElement('li');
    li.className = 'file-item';
    li.innerHTML = `
      <span>${file.name}</span>
      <button type="button" data-index="${index}" aria-label="Remove ${file.name}">✕</button>
    `;
    fileList.appendChild(li);
  });
}

function addFiles(newFiles) {
  for (const file of newFiles) {
    if (!file || !file.name) continue;
    files.push(file);
  }
  renderFileList();
}

function removeFile(index) {
  files.splice(index, 1);
  renderFileList();
}

fileList.addEventListener('click', (event) => {
  const target = event.target;
  if (target.matches('button[data-index]')) {
    const index = Number(target.dataset.index);
    removeFile(index);
  }
});

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (event) => {
  event.preventDefault();
  dropZone.classList.remove('drag-over');
  addFiles(event.dataTransfer.files);
});

fileInput.addEventListener('change', () => {
  addFiles(fileInput.files);
  fileInput.value = '';
});

function updateStatus(status) {
  statusIndicator.textContent = status.charAt(0).toUpperCase() + status.slice(1);
  statusIndicator.className = `status status-${status}`;
  if (status === 'starting' || status === 'running') {
    startButton.setAttribute('disabled', 'disabled');
  } else {
    startButton.removeAttribute('disabled');
  }
}

function appendLogLine(line) {
  logLines.push(line);
  if (logLines.length > 500) {
    logLines = logLines.slice(-500);
  }
  logOutput.textContent = logLines.join('\n');
  logOutput.scrollTop = logOutput.scrollHeight;
}

function replaceLogs(lines) {
  logLines = lines.slice(-500);
  logOutput.textContent = logLines.join('\n');
  logOutput.scrollTop = logOutput.scrollHeight;
}

function syncMetrics(noise, points) {
  const dataset = noise === 'high' ? chart.data.datasets[0] : chart.data.datasets[1];
  dataset.data = points.map((point) => ({ x: point.step, y: point.loss }));
  chart.update('none');

  const latest = points[points.length - 1];
  if (latest) {
    updateMetricSummary(noise, latest.step, latest.loss);
  }
}

function updateMetricSummary(noise, step, loss) {
  const stepEl = noise === 'high' ? highStepEl : lowStepEl;
  const lossEl = noise === 'high' ? highLossEl : lowLossEl;
  stepEl.textContent = step ?? '–';
  lossEl.textContent = typeof loss === 'number' ? loss.toFixed(4) : '–';
}

function addMetricPoint(noise, step, loss) {
  const dataset = noise === 'high' ? chart.data.datasets[0] : chart.data.datasets[1];
  dataset.data.push({ x: step, y: loss });
  dataset.data.sort((a, b) => a.x - b.x);
  chart.update('none');
  updateMetricSummary(noise, step, loss);
}

async function submitForm(event) {
  event.preventDefault();
  if (!files.length) {
    alert('Please add at least one file to train with.');
    return;
  }

  const currentStatus = statusIndicator.textContent.toLowerCase();
  if (currentStatus === 'running' || currentStatus === 'starting') {
    alert('Training is already running.');
    return;
  }

  const formEntries = new FormData(form);
  const payload = new FormData();
  for (const [key, value] of formEntries.entries()) {
    if (key === 'files') continue;
    payload.append(key, value);
  }
  for (const file of files) {
    payload.append('files', file, file.name);
  }

  startButton.setAttribute('disabled', 'disabled');
  updateStatus('starting');

  try {
    const response = await fetch('/train', {
      method: 'POST',
      body: payload,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Failed to start training');
    }
    appendLogLine(`UI: Training started (dataset config: ${data.dataset_config})`);
  } catch (error) {
    console.error(error);
    alert(error.message);
    updateStatus('idle');
  }
}

form.addEventListener('submit', submitForm);

renderFileList();

const events = new EventSource('/events');

events.addEventListener('error', () => {
  statusIndicator.classList.add('status-warning');
});

events.onmessage = (event) => {
  try {
    const data = JSON.parse(event.data);
    switch (data.type) {
      case 'status':
        updateStatus(data.status);
        break;
      case 'log':
        appendLogLine(data.line);
        break;
      case 'log-batch':
        replaceLogs(data.lines);
        break;
      case 'metrics-batch':
        syncMetrics(data.noise, data.points);
        break;
      case 'metric':
        addMetricPoint(data.noise, data.step, data.loss);
        break;
      default:
        break;
    }
  } catch (error) {
    console.error('Failed to parse SSE payload', error);
  }
};
