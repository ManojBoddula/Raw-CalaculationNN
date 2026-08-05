import { client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById("sketchpad");
    const ctx = canvas.getContext("2d");
    const clearBtn = document.getElementById("clear-btn");
    const computeBtn = document.getElementById("compute-btn");
    const svgContainer = document.getElementById("svg-container");
    const statusLog = document.getElementById("status-log");

    // Initialize Canvas (Fill with white background so it's not transparent)
    function initCanvas() {
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.strokeStyle = "#000000"; // Draw in black
        ctx.lineWidth = 12; // Thick brush for CNN readability
    }

    initCanvas();

    // Drawing Logic
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    function getMousePos(e) {
        const rect = canvas.getBoundingClientRect();
        // Handle touch events as well
        if (e.touches && e.touches.length > 0) {
            return {
                x: (e.touches[0].clientX - rect.left) * (canvas.width / rect.width),
                y: (e.touches[0].clientY - rect.top) * (canvas.height / rect.height)
            };
        }
        return {
            x: (e.clientX - rect.left) * (canvas.width / rect.width),
            y: (e.clientY - rect.top) * (canvas.height / rect.height)
        };
    }

    function startDrawing(e) {
        e.preventDefault();
        isDrawing = true;
        const pos = getMousePos(e);
        lastX = pos.x;
        lastY = pos.y;
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const pos = getMousePos(e);
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        
        lastX = pos.x;
        lastY = pos.y;
    }

    function stopDrawing() {
        isDrawing = false;
    }

    // Event Listeners
    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseout", stopDrawing);

    canvas.addEventListener("touchstart", startDrawing, {passive: false});
    canvas.addEventListener("touchmove", draw, {passive: false});
    canvas.addEventListener("touchend", stopDrawing);

    // Clear Button
    clearBtn.addEventListener("click", () => {
        initCanvas();
        svgContainer.innerHTML = '<div class="empty-state">Draw an equation above and click compute.</div>';
        statusLog.style.display = "none";
    });

    // Fetch Metrics on Load
    fetch("/metrics")
        .then(res => res.json())
        .then(data => {
            document.getElementById("v-train-acc").innerText = data.v_train_acc;
            document.getElementById("v-test-acc").innerText = data.v_test_acc;
            document.getElementById("m-train-loss").innerText = data.m_train_loss;
            document.getElementById("m-test-loss").innerText = data.m_test_loss;
            
            const vPlot = document.getElementById("vision-plot");
            const mPlot = document.getElementById("math-plot");
            
            if (data.vision_plot && data.vision_plot.length > 50) {
                vPlot.src = data.vision_plot;
                vPlot.style.display = "block";
            }
            if (data.math_plot && data.math_plot.length > 50) {
                mPlot.src = data.math_plot;
                mPlot.style.display = "block";
            }
        })
        .catch(err => console.error("Error fetching metrics:", err));

    // Compute Button
    computeBtn.addEventListener("click", async () => {
        const base64Image = canvas.toDataURL("image/png");
        
        computeBtn.disabled = true;
        computeBtn.innerText = "Computing... ⏳";
        svgContainer.innerHTML = '<div class="empty-state">Processing neural logic...</div>';
        statusLog.style.display = "none";
        statusLog.className = "status-box";

        try {
            // Connect to Hugging Face ZeroGPU Gradio App
            const hf_client = await client("Manoj8179/Neural-Cal");
            
            // Send the canvas image to the /predict endpoint (which triggers master_execution_flow)
            const result = await hf_client.predict("/predict", [
                { "background": null, "composite": base64Image, "layers": [] }
            ]);

            const data = {
                svg: result.data[0],
                status: result.data[3]
            };

            // Display result
            svgContainer.innerHTML = data.svg;
            statusLog.innerText = data.status;
            statusLog.style.display = "block";
            
            // Play Audio via Web Speech API (Optional parsing of status)
            try {
                const phrase = data.status.split("\\n")[1];
                if (phrase) {
                    const utterance = new SpeechSynthesisUtterance(phrase);
                    utterance.rate = 1.0;
                    window.speechSynthesis.speak(utterance);
                }
            } catch (e) {
                console.log("Audio synthesis not supported or blocked by browser.");
            }

            computeBtn.disabled = false;
            computeBtn.innerText = "Compute Formula 🚀";
        } catch (error) {
            console.error("Error computing formula:", error);
            statusLog.innerText = `❌ Error: ${error.message}`;
            statusLog.style.display = "block";
            statusLog.className = "status-box error";
            computeBtn.disabled = false;
            computeBtn.innerText = "Compute Formula 🚀";
        }
    });
});
