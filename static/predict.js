const form = document.getElementById('prediction-form');
const submitBtn = document.getElementById('submit-btn');
const btnText = submitBtn.querySelector('.btn-text');
const spinner = submitBtn.querySelector('.spinner');
const resultContainer = document.getElementById('result-container');
const errorContainer = document.getElementById('error-container');
const loadingOverlay = document.getElementById('loading-overlay');

// Loading messages that cycle during the animation
const loadingMessages = [
    'Retrieving features...',
    'Running XGBoost inference...',
    'Calculating lag variables...',
    'Finalising prediction...',
];

let messageInterval = null;

function showLoading() {
    let i = 0;
    const msgEl = document.getElementById('loading-msg');
    msgEl.textContent = loadingMessages[0];
    loadingOverlay.classList.remove('hidden');

    messageInterval = setInterval(() => {
        i = (i + 1) % loadingMessages.length;
        msgEl.style.opacity = '0';
        setTimeout(() => {
            msgEl.textContent = loadingMessages[i];
            msgEl.style.opacity = '1';
        }, 200);
    }, 900);
}

function hideLoading() {
    clearInterval(messageInterval);
    loadingOverlay.classList.add('hidden');
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const shopId = document.getElementById('shop_id').value;
    const itemId = document.getElementById('item_id').value;

    // Loading state
    btnText.classList.add('hidden');
    spinner.classList.remove('hidden');
    submitBtn.disabled = true;
    resultContainer.classList.add('hidden');
    errorContainer.classList.add('hidden');

    showLoading();

    // Minimum visual loading time: random 3–5 seconds
    const minDelay = 3000 + Math.random() * 2000;
    const startTime = Date.now();

    try {
        const [response] = await Promise.all([
            fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ shop_id: shopId, item_id: itemId })
            }),
            sleep(minDelay)
        ]);

        const data = await response.json();

        hideLoading();

        if (response.ok) {
            document.getElementById('prediction-value').innerText = data.predicted_sales;
            document.getElementById('result-shop').innerText = data.shop_id;
            document.getElementById('result-item').innerText = data.item_id;
            resultContainer.classList.remove('hidden');
        } else {
            document.getElementById('error-message').innerText = data.error || 'An unexpected error occurred.';
            errorContainer.classList.remove('hidden');
        }
    } catch (err) {
        hideLoading();
        document.getElementById('error-message').innerText = 'Unable to reach the server. Make sure app.py is running.';
        errorContainer.classList.remove('hidden');
    } finally {
        btnText.classList.remove('hidden');
        spinner.classList.add('hidden');
        submitBtn.disabled = false;
    }
});
