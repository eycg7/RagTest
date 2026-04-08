import os
import json
import requests
from flask import Flask, request, jsonify, render_template_string, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'gemma2')
UPLOAD_FOLDER = '/tmp/rag_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.doc', '.md', '.csv', '.json', '.html'}


def extract_text(filepath: str, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()

    if ext == '.pdf':
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                return '\n'.join(page.extract_text() or '' for page in pdf.pages)
        except ImportError:
            try:
                import pypdf
                reader = pypdf.PdfReader(filepath)
                return '\n'.join(page.extract_text() or '' for page in reader.pages)
            except ImportError:
                return '[PDF extraction unavailable: install pdfplumber or pypdf]'

    if ext in ('.docx', '.doc'):
        try:
            import docx
            doc = docx.Document(filepath)
            return '\n'.join(p.text for p in doc.paragraphs)
        except ImportError:
            return '[DOCX extraction unavailable: install python-docx]'

    # Plain text formats
    for encoding in ('utf-8', 'latin-1', 'cp1252'):
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, OSError):
            continue

    return '[Could not decode file content]'


def summarize_with_ollama(filename: str, text: str) -> str:
    # Truncate very large documents to avoid context limits
    max_chars = 12000
    truncated = text[:max_chars]
    if len(text) > max_chars:
        truncated += f'\n\n[... document truncated at {max_chars} characters ...]'

    prompt = (
        f'You are a document analyst. Below is the content of a file named "{filename}".\n'
        f'Write a concise but thorough summary of this document. '
        f'Include the main topics, key points, and any important details.\n\n'
        f'Document content:\n{truncated}\n\n'
        f'Summary of "{filename}":'
    )

    response = requests.post(
        f'{OLLAMA_URL}/api/generate',
        json={
            'model': OLLAMA_MODEL,
            'prompt': prompt,
            'stream': False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get('response', '').strip()


HTML_PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Document Summarizer – gemma2</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f172a; color: #e2e8f0; min-height: 100vh;
    display: flex; flex-direction: column; align-items: center; padding: 2rem 1rem;
  }
  h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.25rem; color: #f8fafc; }
  .subtitle { color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem; }
  .model-badge {
    display: inline-block; background: #1e3a5f; border: 1px solid #3b82f6;
    color: #93c5fd; border-radius: 9999px; padding: 0.15rem 0.75rem;
    font-size: 0.75rem; font-weight: 600; margin-bottom: 2rem;
  }
  #drop-zone {
    width: 100%; max-width: 700px; border: 2px dashed #334155;
    border-radius: 1rem; padding: 3rem 2rem; text-align: center;
    cursor: pointer; transition: all 0.2s; background: #1e293b;
  }
  #drop-zone.dragover {
    border-color: #3b82f6; background: #1e3a5f; color: #93c5fd;
  }
  #drop-zone svg { width: 48px; height: 48px; color: #475569; margin-bottom: 1rem; }
  #drop-zone p { color: #94a3b8; font-size: 0.95rem; }
  #drop-zone strong { color: #e2e8f0; }
  #file-input { display: none; }
  .file-types { font-size: 0.78rem; color: #64748b; margin-top: 0.5rem; }
  #file-list {
    width: 100%; max-width: 700px; margin-top: 1.5rem;
    display: flex; flex-direction: column; gap: 0.5rem;
  }
  .file-chip {
    background: #1e293b; border: 1px solid #334155; border-radius: 0.5rem;
    padding: 0.5rem 0.75rem; display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.85rem;
  }
  .file-chip .name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .file-chip .size { color: #64748b; font-size: 0.75rem; white-space: nowrap; }
  .remove-btn {
    background: none; border: none; color: #ef4444; cursor: pointer;
    font-size: 1rem; line-height: 1; padding: 0 0.25rem;
  }
  #summarize-btn {
    margin-top: 1.5rem; padding: 0.75rem 2.5rem;
    background: #3b82f6; color: #fff; border: none;
    border-radius: 0.5rem; font-size: 1rem; font-weight: 600;
    cursor: pointer; transition: background 0.2s;
    width: 100%; max-width: 700px;
  }
  #summarize-btn:hover:not(:disabled) { background: #2563eb; }
  #summarize-btn:disabled { background: #334155; color: #64748b; cursor: not-allowed; }
  #results { width: 100%; max-width: 700px; margin-top: 2rem; display: flex; flex-direction: column; gap: 1.5rem; }
  .result-card {
    background: #1e293b; border: 1px solid #334155; border-radius: 1rem;
    overflow: hidden;
  }
  .result-card.success { border-color: #334155; }
  .result-card.error { border-color: #ef4444; }
  .result-card.loading { border-color: #3b82f6; }
  .card-header {
    padding: 0.75rem 1rem; background: #0f172a;
    display: flex; align-items: center; gap: 0.5rem;
  }
  .card-header .filename { font-weight: 600; font-size: 0.95rem; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .status-badge {
    font-size: 0.7rem; padding: 0.15rem 0.5rem; border-radius: 9999px; font-weight: 600;
  }
  .status-badge.loading { background: #1e3a5f; color: #93c5fd; }
  .status-badge.success { background: #052e16; color: #4ade80; }
  .status-badge.error { background: #450a0a; color: #f87171; }
  .card-body { padding: 1rem; font-size: 0.9rem; line-height: 1.7; color: #cbd5e1; white-space: pre-wrap; word-break: break-word; }
  .spinner {
    width: 16px; height: 16px; border: 2px solid #3b82f6;
    border-top-color: transparent; border-radius: 50%;
    animation: spin 0.7s linear infinite; display: inline-block;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .overall-progress { width: 100%; max-width: 700px; margin-top: 1rem; }
  .progress-bar-bg { background: #1e293b; border-radius: 9999px; height: 6px; overflow: hidden; }
  .progress-bar { background: #3b82f6; height: 100%; transition: width 0.4s; border-radius: 9999px; }
  .progress-label { font-size: 0.78rem; color: #64748b; margin-bottom: 0.25rem; text-align: right; }
</style>
</head>
<body>
<h1>Document Summarizer</h1>
<div class="subtitle">Drag &amp; drop documents — powered by Ollama</div>
<span class="model-badge">model: gemma2</span>

<div id="drop-zone">
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
      d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
  </svg>
  <p><strong>Drop files here</strong> or click to browse</p>
  <p class="file-types">Supported: .txt .pdf .docx .md .csv .json .html</p>
</div>
<input type="file" id="file-input" multiple
  accept=".txt,.pdf,.docx,.doc,.md,.csv,.json,.html">

<div id="file-list"></div>

<button id="summarize-btn" disabled>Summarize Documents</button>

<div class="overall-progress" id="progress-wrap" style="display:none">
  <div class="progress-label" id="progress-label">0 / 0</div>
  <div class="progress-bar-bg"><div class="progress-bar" id="progress-bar" style="width:0%"></div></div>
</div>

<div id="results"></div>

<script>
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileList = document.getElementById('file-list');
const summarizeBtn = document.getElementById('summarize-btn');
const results = document.getElementById('results');
const progressWrap = document.getElementById('progress-wrap');
const progressBar = document.getElementById('progress-bar');
const progressLabel = document.getElementById('progress-label');

let files = [];

function fmtSize(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
  return (b/1048576).toFixed(1) + ' MB';
}

function renderFileList() {
  fileList.innerHTML = '';
  files.forEach((f, i) => {
    const chip = document.createElement('div');
    chip.className = 'file-chip';
    chip.innerHTML = `
      <span class="name" title="${f.name}">${f.name}</span>
      <span class="size">${fmtSize(f.size)}</span>
      <button class="remove-btn" data-i="${i}" title="Remove">&#x2715;</button>`;
    fileList.appendChild(chip);
  });
  summarizeBtn.disabled = files.length === 0;
}

function addFiles(newFiles) {
  for (const f of newFiles) {
    if (!files.find(x => x.name === f.name && x.size === f.size)) files.push(f);
  }
  renderFileList();
}

fileList.addEventListener('click', e => {
  const btn = e.target.closest('.remove-btn');
  if (!btn) return;
  files.splice(+btn.dataset.i, 1);
  renderFileList();
});

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { addFiles(fileInput.files); fileInput.value = ''; });

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  addFiles(e.dataTransfer.files);
});

async function summarizeFile(file, card, bodyEl, badgeEl) {
  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch('/summarize', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Server error');
    card.classList.remove('loading'); card.classList.add('success');
    badgeEl.className = 'status-badge success'; badgeEl.textContent = 'done';
    bodyEl.textContent = data.summary;
  } catch (err) {
    card.classList.remove('loading'); card.classList.add('error');
    badgeEl.className = 'status-badge error'; badgeEl.textContent = 'error';
    bodyEl.textContent = 'Error: ' + err.message;
  }
}

summarizeBtn.addEventListener('click', async () => {
  if (files.length === 0) return;
  results.innerHTML = '';
  summarizeBtn.disabled = true;
  progressWrap.style.display = 'block';

  const total = files.length;
  let done = 0;

  function updateProgress() {
    progressLabel.textContent = `${done} / ${total}`;
    progressBar.style.width = (done / total * 100) + '%';
  }
  updateProgress();

  // Create all cards up front
  const cards = files.map(file => {
    const card = document.createElement('div');
    card.className = 'result-card loading';
    card.innerHTML = `
      <div class="card-header">
        <span class="filename" title="${file.name}">${file.name}</span>
        <span class="spinner"></span>
        <span class="status-badge loading">processing</span>
      </div>
      <div class="card-body">Sending to gemma2…</div>`;
    results.appendChild(card);
    const body = card.querySelector('.card-body');
    const badge = card.querySelector('.status-badge');
    const spinner = card.querySelector('.spinner');
    return { card, body, badge, spinner };
  });

  // Process all files concurrently
  await Promise.all(files.map(async (file, i) => {
    const { card, body, badge, spinner } = cards[i];
    await summarizeFile(file, card, body, badge);
    spinner.style.display = 'none';
    done++;
    updateProgress();
  }));

  summarizeBtn.disabled = false;
});
</script>
</body>
</html>'''


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/summarize', methods=['POST'])
def summarize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['file']
    filename = secure_filename(f.filename)
    if not filename:
        return jsonify({'error': 'Invalid filename'}), 400

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    f.save(filepath)

    try:
        text = extract_text(filepath, filename)
        if not text.strip():
            return jsonify({'error': 'Could not extract text from document'}), 400

        summary = summarize_with_ollama(filename, text)
        return jsonify({'filename': filename, 'summary': summary})

    except requests.exceptions.ConnectionError:
        return jsonify({'error': f'Cannot connect to Ollama at {OLLAMA_URL}. Is Ollama running?'}), 503
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Ollama timed out. The document may be too large.'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/download')
def download_zip():
    import zipfile
    zip_path = '/tmp/RagTest.zip'
    project_dir = os.path.dirname(os.path.abspath(__file__))
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in ('app.py', 'requirements.txt', 'README.md'):
            fpath = os.path.join(project_dir, fname)
            if os.path.exists(fpath):
                zf.write(fpath, fname)
    return send_file(zip_path, as_attachment=True, download_name='RagTest.zip')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
