/* Frontend JavaScript for Ancient Script Decoder */

const API_BASE = window.location.origin;
let selectedFile = null;
let currentResults = null;

// Initialize drag and drop
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    // Drag over
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    // Drag leave
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    // Drop
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    // Click upload area
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
});

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }
    
    selectedFile = file;
    document.getElementById('processBtn').disabled = false;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const resultSection = document.getElementById('resultsSection');
        resultSection.style.display = 'block';
        document.getElementById('originalImage').src = e.target.result;
        
        // Scroll to results
        resultSection.scrollIntoView({ behavior: 'smooth' });
    };
    reader.readAsDataURL(file);
}

async function processImage() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }
    
    const sourceLanguage = document.getElementById('sourceLanguage').value;
    const targetLanguage = document.getElementById('targetLanguage').value;
    
    // Show loading
    showLoading('Processing image through the AI pipeline...');
    
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
        
    } catch (error) {
        hideLoading();
        showError(`Error: ${error.message}`);
    }
}

function displayResults(results) {
    const pipeline = results.pipeline;
    const finalOutput = results.final_output;
    
    // Display images
    if (pipeline.preprocessing.image) {
        document.getElementById('preprocessedImage').src = 
            `data:image/png;base64,${pipeline.preprocessing.image}`;
    }
    
    if (pipeline.detection.image) {
        document.getElementById('detectionImage').src = 
            `data:image/png;base64,${pipeline.detection.image}`;
    }
    
    // Display text results
    document.getElementById('rawText').textContent = 
        pipeline.recognition.raw_text || 'No text detected';
    
    document.getElementById('cleanedText').textContent = 
        pipeline.cleaning.cleaned_text || 'No text after cleaning';
    
    document.getElementById('translatedText').textContent = 
        finalOutput.translated_text || pipeline.cleaning.cleaned_text;
    
    // Display statistics
    const statsGrid = document.getElementById('statsGrid');
    statsGrid.innerHTML = '';
    
    const stats = [
        { label: 'Regions Detected', value: pipeline.detection.regions_found },
        { label: 'Original Length', value: pipeline.cleaning.original_length },
        { label: 'Cleaned Length', value: pipeline.cleaning.cleaned_length },
        { label: 'Confidence Score', value: (finalOutput.confidence_score * 100).toFixed(1) + '%' }
    ];
    
    stats.forEach(stat => {
        const statItem = document.createElement('div');
        statItem.className = 'stat-item';
        statItem.innerHTML = `
            <div class="stat-label">${stat.label}</div>
            <div class="stat-value">${stat.value}</div>
        `;
        statsGrid.appendChild(statItem);
    });
    
    console.log('Results displayed successfully');
}

function downloadResults() {
    if (!currentResults) return;
    
    const dataStr = JSON.stringify(currentResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `ancient_script_results_${new Date().toISOString()}.json`;
    link.click();
}

function resetForm() {
    selectedFile = null;
    currentResults = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('processBtn').disabled = true;
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('uploadArea').classList.remove('drag-over');
}

function showLoading(message = 'Processing...') {
    document.getElementById('loadingSpinner').style.display = 'flex';
    document.getElementById('loadingText').textContent = message;
    document.getElementById('errorMessage').style.display = 'none';
}

function hideLoading() {
    document.getElementById('loadingSpinner').style.display = 'none';
}

function showError(message) {
    document.getElementById('errorMessage').style.display = 'flex';
    document.getElementById('errorText').textContent = message;
}

function closeError() {
    document.getElementById('errorMessage').style.display = 'none';
}

// Handle API errors
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

// Check API health on page load
window.addEventListener('load', async () => {
    const isHealthy = await checkApiHealth();
    if (!isHealthy) {
        showError('Warning: Backend API is not responding. Please ensure the server is running on http://localhost:8000');
    }
});
