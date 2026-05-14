/* Frontend JavaScript for Ancient Script Decoder */

const API_BASE = window.location.origin;
let selectedFile = null;
let currentResults = null;

const loadingMessages = [
    'Calibrating manuscript light...',
    'Enhancing faded ink patterns...',
    'Locating text regions...',
    'Reading ancient glyph forms...',
    'Cleaning recovered text...',
    'Preparing translation...'
];
let loadingMessageTimer = null;

// Initialize drag and drop
// Kept dependency-free so the FastAPI static frontend stays simple to serve.
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    loadSupportedLanguages();

    document.querySelectorAll('.reveal-card').forEach((card, index) => {
        card.style.animationDelay = `${index * 90}ms`;
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
});

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file. Ancient stone tablets are welcome, spreadsheets are not.');
        return;
    }

    selectedFile = file;

    const processBtn = document.getElementById('processBtn');
    const uploadArea = document.getElementById('uploadArea');
    const uploadTitle = document.getElementById('uploadTitle');
    const uploadHint = document.getElementById('uploadHint');

    processBtn.disabled = false;
    uploadArea.classList.add('has-file');
    uploadTitle.textContent = file.name;
    uploadHint.textContent = `${formatFileSize(file.size)} selected. Ready to decode.`;
    closeError();

    const reader = new FileReader();
    reader.onload = (e) => {
        const resultSection = document.getElementById('resultsSection');
        resultSection.style.display = 'block';
        resultSection.classList.add('reveal-card');
        document.getElementById('originalImage').src = e.target.result;
        clearPreviousOutputs();
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
    reader.readAsDataURL(file);
}

async function processImage() {
    if (!selectedFile) {
        showError('Please select an image first. The decoder needs an artifact before it can do its magic.');
        return;
    }

    const sourceLanguage = document.getElementById('sourceLanguage').value;
    const targetLanguage = document.getElementById('targetLanguage').value;
    const processBtn = document.getElementById('processBtn');

    processBtn.disabled = true;
    showLoading('Calibrating manuscript light...');

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('source_language', sourceLanguage);
        formData.append('target_language', targetLanguage);

        const response = await fetch(`${API_BASE}/process`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Processing failed');
        }

        const results = await response.json();
        currentResults = results;

        displayResults(results);
        hideLoading();
        processBtn.disabled = false;
    } catch (error) {
        hideLoading();
        processBtn.disabled = false;
        showError(`Error: ${error.message}`);
    }
}

function displayResults(results) {
    const pipeline = results.pipeline;
    const finalOutput = results.final_output;
    const resultsSection = document.getElementById('resultsSection');

    resultsSection.style.display = 'block';

    if (pipeline.preprocessing.image) {
        document.getElementById('preprocessedImage').src =
            `data:image/png;base64,${pipeline.preprocessing.image}`;
    }

    if (pipeline.detection.image) {
        document.getElementById('detectionImage').src =
            `data:image/png;base64,${pipeline.detection.image}`;
    }

    document.getElementById('rawText').textContent =
        pipeline.recognition.raw_text || 'No text detected';

    document.getElementById('cleanedText').textContent =
        pipeline.cleaning.cleaned_text || 'No text after cleaning';

    document.getElementById('translatedText').textContent =
        finalOutput.translated_text || pipeline.cleaning.cleaned_text || 'No translation available';

    const statsGrid = document.getElementById('statsGrid');
    statsGrid.innerHTML = '';

    const confidence = Number(finalOutput.confidence_score || 0);
    const stats = [
        { label: 'Regions Detected', value: pipeline.detection.regions_found ?? 0 },
        { label: 'Original Length', value: pipeline.cleaning.original_length ?? 0 },
        { label: 'Cleaned Length', value: pipeline.cleaning.cleaned_length ?? 0 },
        { label: 'Confidence Score', value: `${(confidence * 100).toFixed(1)}%` }
    ];

    stats.forEach((stat, index) => {
        const statItem = document.createElement('div');
        statItem.className = 'stat-item';
        statItem.style.animationDelay = `${index * 70}ms`;
        statItem.innerHTML = `
            <div class="stat-label">${stat.label}</div>
            <div class="stat-value">${stat.value}</div>
        `;
        statsGrid.appendChild(statItem);
    });

    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function downloadResults() {
    if (!currentResults) {
        showError('No results to download yet. Run the decoder first.');
        return;
    }

    const dataStr = JSON.stringify(currentResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `ancient_script_results_${new Date().toISOString()}.json`;
    link.click();
    URL.revokeObjectURL(url);
}

function resetForm() {
    selectedFile = null;
    currentResults = null;

    const uploadArea = document.getElementById('uploadArea');
    document.getElementById('fileInput').value = '';
    document.getElementById('processBtn').disabled = true;
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('uploadTitle').textContent = 'Drop your inscription here';
    document.getElementById('uploadHint').textContent = 'PNG, JPG, WEBP, or any image format your browser supports';
    uploadArea.classList.remove('drag-over', 'has-file');
    clearPreviousOutputs();
    closeError();
    document.getElementById('processPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function clearPreviousOutputs() {
    document.getElementById('preprocessedImage').removeAttribute('src');
    document.getElementById('detectionImage').removeAttribute('src');
    document.getElementById('rawText').textContent = 'Ready for OCR extraction.';
    document.getElementById('cleanedText').textContent = 'Cleaned text will appear here.';
    document.getElementById('translatedText').textContent = 'Translation will appear here.';
    document.getElementById('statsGrid').innerHTML = '';
}

function showLoading(message = 'Processing...') {
    const loadingSpinner = document.getElementById('loadingSpinner');
    const loadingText = document.getElementById('loadingText');
    let messageIndex = 0;

    loadingSpinner.style.display = 'flex';
    loadingText.textContent = message;
    document.getElementById('errorMessage').style.display = 'none';

    clearInterval(loadingMessageTimer);
    loadingMessageTimer = setInterval(() => {
        messageIndex = (messageIndex + 1) % loadingMessages.length;
        loadingText.textContent = loadingMessages[messageIndex];
    }, 1450);

    loadingSpinner.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function hideLoading() {
    clearInterval(loadingMessageTimer);
    loadingMessageTimer = null;
    document.getElementById('loadingSpinner').style.display = 'none';
}

function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.style.display = 'flex';
    document.getElementById('errorText').textContent = message;
    errorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function closeError() {
    document.getElementById('errorMessage').style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) {
            console.warn('Backend API not ready yet');
            return false;
        }
        return true;
    } catch (error) {
        console.error('Cannot connect to backend API:', error);
        return false;
    }
}

async function loadSupportedLanguages() {
    try {
        const response = await fetch(`${API_BASE}/languages`);
        if (!response.ok) return;

        const data = await response.json();
        const sourceSelect = document.getElementById('sourceLanguage');
        const targetSelect = document.getElementById('targetLanguage');

        if (Array.isArray(data.source_languages) && sourceSelect) {
            sourceSelect.innerHTML = '';
            data.source_languages.forEach((language) => {
                const option = document.createElement('option');
                option.value = language.code;
                option.textContent = language.label || language.code;
                option.title = language.note || '';
                if (language.code === (data.default_source || 'la')) {
                    option.selected = true;
                }
                sourceSelect.appendChild(option);
            });
        }

        if (Array.isArray(data.target_languages) && targetSelect) {
            targetSelect.innerHTML = '';
            data.target_languages.forEach((language) => {
                const option = document.createElement('option');
                option.value = language.code;
                option.textContent = language.label || language.code;
                if (language.code === (data.default_target || 'en')) {
                    option.selected = true;
                }
                targetSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.warn('Could not load dynamic language list:', error);
    }
}

window.addEventListener('load', async () => {
    const isHealthy = await checkApiHealth();
    if (!isHealthy) {
        showError('Backend API is not responding. Please make sure the FastAPI server is running before processing an image.');
    }
});
