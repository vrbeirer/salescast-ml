document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const shopId = document.getElementById('shop_id').value;
    const itemId = document.getElementById('item_id').value;
    
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner');
    
    const resultContainer = document.getElementById('result-container');
    const errorContainer = document.getElementById('error-container');
    
    // UI state loading
    btnText.classList.add('hidden');
    spinner.classList.remove('hidden');
    submitBtn.disabled = true;
    resultContainer.classList.add('hidden');
    errorContainer.classList.add('hidden');
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ shop_id: shopId, item_id: itemId })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            document.getElementById('prediction-value').innerText = data.predicted_sales;
            resultContainer.classList.remove('hidden');
        } else {
            document.getElementById('error-message').innerText = data.error || 'An error occurred';
            errorContainer.classList.remove('hidden');
        }
    } catch (err) {
        document.getElementById('error-message').innerText = 'Failed to connect to the server.';
        errorContainer.classList.remove('hidden');
    } finally {
        // Restore UI
        btnText.classList.remove('hidden');
        spinner.classList.add('hidden');
        submitBtn.disabled = false;
    }
});
