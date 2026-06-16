document.addEventListener('DOMContentLoaded', () => {
    const skillsInput = document.getElementById('skillsInput');
    const charCount = document.getElementById('charCount');
    const predictBtn = document.getElementById('predictBtn');
    
    // Result sections
    const initialState = document.getElementById('initialState');
    const predictionState = document.getElementById('predictionState');
    
    const predictedDomain = document.getElementById('predictedDomain');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceBar = document.getElementById('confidenceBar');
    const chipsContainer = document.getElementById('chipsContainer');

    // Update character count
    skillsInput.addEventListener('input', () => {
        const count = skillsInput.value.length;
        charCount.textContent = `${count} / 2000 characters`;
    });

    // Handle prediction
    predictBtn.addEventListener('click', async () => {
        const text = skillsInput.value.trim();
        
        if (!text) {
            alert('Please enter some skills or resume content');
            skillsInput.focus();
            return;
        }

        // Show loading state
        const originalBtnText = predictBtn.innerHTML;
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<svg class="spinner" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle; margin-right:8px; animation: spin 1s linear infinite;"><line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line><line x1="2" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="22" y2="12"></line><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line></svg> Predicting...';
        
        // Ensure UI resets while fetching
        predictionState.style.display = 'none';
        initialState.style.display = 'flex';
        confidenceBar.style.width = '0%';

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                throw new Error('Prediction failed');
            }

            const data = await response.json();
            
            // Update UI with results
            predictedDomain.textContent = data.prediction;
            confidenceValue.textContent = `${data.confidence}%`;
            
            const topMatches = data.top_matches || [data.prediction, "Data Analyst", "Machine Learning Engineer", "AI Engineer"];
            
            chipsContainer.innerHTML = '';
            topMatches.forEach(match => {
                const chip = document.createElement('span');
                chip.className = 'chip';
                chip.textContent = match;
                chipsContainer.appendChild(chip);
            });
            
            // Toggle states
            initialState.style.display = 'none';
            predictionState.style.display = 'block';
            
            // Animate progress bar
            setTimeout(() => {
                confidenceBar.style.width = `${data.confidence}%`;
            }, 50);

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while making the prediction. Is the backend running?');
        } finally {
            // Reset button
            predictBtn.disabled = false;
            predictBtn.innerHTML = originalBtnText;
        }
    });
});