const submitBtn = document.getElementById('submitBtn');
const loading = document.getElementById('loading');
const resultDiv = document.getElementById('result');
const clearBtn = document.getElementById('clearBtn');
const lyricsInput = document.getElementById('lyrics');
const modelRadios = document.querySelectorAll('.model-selection');

function checkForm() {
    const lyrics = lyricsInput.value.trim();
    const modelSelected = Array.from(modelRadios).some(radio => radio.checked);
    submitBtn.disabled = !(lyrics && modelSelected);
}

lyricsInput.addEventListener('input', checkForm);
modelRadios.forEach(radio => radio.addEventListener('change', checkForm));

submitBtn.addEventListener('click', async () => {
    const selectedModel = document.querySelector('input[name="model"]:checked').value;
    const lyrics = lyricsInput.value.trim();
    
    submitBtn.classList.add('hidden');
    loading.classList.remove('hidden');
    resultDiv.classList.add('hidden');
    
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel, lyrics: lyrics })
    });
    
    const resp = await response.json();
    document.getElementById('predictionLabel').textContent = resp.prediction_label;
    document.getElementById('predictionScore').textContent = resp.prediction_score;
    document.getElementById('barChart').src = resp.bar_chart;
    document.getElementById('pieChart').src = resp.pie_chart;
    
    loading.classList.add('hidden');
    submitBtn.classList.remove('hidden');
    resultDiv.classList.remove('hidden');
});

clearBtn.addEventListener('click', () => {
    lyricsInput.value = '';
    modelRadios.forEach(radio => radio.checked = false);
    resultDiv.classList.add('hidden');
    checkForm();
});
