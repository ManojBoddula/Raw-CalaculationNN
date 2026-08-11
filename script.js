document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const svgContainer = document.getElementById('svg-container');
    const resultsSection = document.getElementById('results-section');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');

    // Drawing state
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let currentColor = 'white';
    let isEraser = false;
    
    const toolPen = document.getElementById('tool-pen');
    const toolEraser = document.getElementById('tool-eraser');
    const colorBtns = document.querySelectorAll('.color-btn');

    // Tool selection
    toolPen.addEventListener('click', () => {
        isEraser = false;
        toolPen.classList.add('active');
        toolEraser.classList.remove('active');
    });

    toolEraser.addEventListener('click', () => {
        isEraser = true;
        toolEraser.classList.add('active');
        toolPen.classList.remove('active');
    });

    colorBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            colorBtns.forEach(b => b.classList.remove('active'));
            const target = e.currentTarget;
            target.classList.add('active');
            currentColor = target.dataset.color;
            // Also switch back to pen automatically if eraser was selected
            if (isEraser) {
                isEraser = false;
                toolPen.classList.add('active');
                toolEraser.classList.remove('active');
            }
        });
    });

    // Resize canvas properly for retina displays while keeping aspect ratio
    function resizeCanvas() {
        const wrapper = document.querySelector('.canvas-wrapper');
        const ratio = Math.max(window.devicePixelRatio || 1, 1);
        canvas.width = wrapper.clientWidth * ratio;
        canvas.height = wrapper.clientHeight * ratio;
        ctx.scale(ratio, ratio);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.clearRect(0, 0, canvas.width, canvas.height); // Keep it transparent
    }

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas(); // Initial setup

    // Drawing Events
    function startDrawing(e) {
        isDrawing = true;
        const pos = getMousePos(e);
        [lastX, lastY] = [pos.x, pos.y];
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(lastX, lastY);
        ctx.stroke();
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault(); // Prevent scrolling on touch
        const pos = getMousePos(e);
        ctx.beginPath();
        
        if (isEraser) {
            ctx.globalCompositeOperation = 'destination-out';
            ctx.lineWidth = 30; // Eraser is thicker
        } else {
            ctx.globalCompositeOperation = 'source-over';
            ctx.strokeStyle = currentColor;
            ctx.lineWidth = 10;
        }

        ctx.moveTo(lastX, lastY);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        [lastX, lastY] = [pos.x, pos.y];
    }

    function stopDrawing() {
        isDrawing = false;
    }

    function getMousePos(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: (clientX - rect.left),
            y: (clientY - rect.top)
        };
    }

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    canvas.addEventListener('touchstart', startDrawing, {passive: false});
    canvas.addEventListener('touchmove', draw, {passive: false});
    canvas.addEventListener('touchend', stopDrawing);

    clearBtn.addEventListener('click', () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        resultsSection.classList.add('hidden');
        svgContainer.innerHTML = `
            <div class="placeholder-text">
                Awaiting inference task.
            </div>
        `;
    });

    // API Call
    analyzeBtn.addEventListener('click', async () => {
        const imageData = canvas.toDataURL('image/png');
        
        loadingOverlay.classList.remove('hidden');
        analyzeBtn.disabled = true;
        resultsSection.classList.add('hidden');
        
        try {
            const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
            const renderUrl = 'https://raw-calaculationnn-1.onrender.com';
            const targetUrl = isLocal ? 'http://127.0.0.1:7860/predict' : renderUrl + '/predict';

            const res = await fetch(targetUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });

            const result = await res.json();

            if (!res.ok || result.error) {
                throw new Error(result.error || result.detail || 'Prediction failed');
            }

            // Animate Steps
            await animateSequence(result);

        } catch (err) {
            console.error(err);
            alert("Error: " + err.message);
            loadingOverlay.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    });

    // We can also fetch metrics on load
    async function fetchMetrics() {
        try {
            const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
            const renderUrl = 'https://raw-calaculationnn-1.onrender.com';
            
            const targetUrl = isLocal ? 'http://127.0.0.1:7860/metrics' : renderUrl + '/metrics';
            
            const res = await fetch(targetUrl);
            const data = await res.json();
            document.getElementById('val-v-acc').textContent = data.v_test_acc;
            document.getElementById('val-m-loss').textContent = data.m_test_loss;
        } catch(e) {
            console.log("Could not fetch metrics", e);
        }
    }
    fetchMetrics();

    // Animation Sequencer
    async function animateSequence(data) {
        // Step 1: Vision
        loadingText.textContent = "Evaluating character strokes...";
        svgContainer.innerHTML = data.svg_step1;
        await sleep(150);

        // Step 2: Bridge
        loadingText.textContent = "Reasoning...";
        svgContainer.innerHTML = data.svg_step2;
        await sleep(150);

        // Step 3: Complete
        loadingText.textContent = `Result: ${data.result}`;
        svgContainer.innerHTML = data.svg_step3;
        await sleep(100);

        // Render Telemetry
        renderTelemetry(data);
        
        loadingOverlay.classList.add('hidden');
        analyzeBtn.disabled = false;
        resultsSection.classList.remove('hidden');
        
        // Speak result
        try {
            const utterance = new SpeechSynthesisUtterance(`The calculated equation result for ${data.equation} is ${data.result}`);
            speechSynthesis.speak(utterance);
        } catch(e) {}
    }

    function renderTelemetry(data) {
        // Gallery
        const gallery = document.getElementById('token-gallery');
        gallery.innerHTML = '';
        data.gallery.forEach(item => {
            const div = document.createElement('div');
            div.className = 'gallery-item';
            div.innerHTML = `
                <img src="data:image/png;base64,${item.image}" alt="token">
                <span>${item.label}</span>
            `;
            gallery.appendChild(div);
        });

        // Chart
        const chart = document.getElementById('activation-chart');
        chart.src = `data:image/png;base64,${data.chart}`;

        // Log
        const log = document.getElementById('execution-log');
        log.textContent = `🎯 NEURAL MATHEMATICAL PIPELINE SUCCESS:\n${data.log}`;
    }

    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
});
