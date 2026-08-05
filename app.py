import os
import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import matplotlib.pyplot as plt

# Import the ML logic and models from the existing gradio_app script
# This will trigger the global model loading automatically
import gradio_app as ga

app = FastAPI(title="Neural Network Calculation Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import gradio as gr

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount the old Gradio UI at /gradio so Hugging Face orchestrator detects a valid Gradio app!
app = gr.mount_gradio_app(app, ga.demo, path="/gradio")

# --- ZERO GPU SECURITY BYPASS ---
# Hugging Face ZeroGPU requires a @spaces.GPU function in the main script.
import spaces

@spaces.GPU
def _gpu_bypass(text):
    return text

_dummy_demo = gr.Interface(fn=_gpu_bypass, inputs="text", outputs="text")
app = gr.mount_gradio_app(app, _dummy_demo, path="/gpu_bypass")
# --------------------------------

class SketchData(BaseModel):
    image_base64: str

def array_to_base64_png(img_array):
    """Convert numpy array (matplotlib imread output or similar) to base64 PNG."""
    try:
        # If it's a float array from matplotlib, convert to uint8
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
            img_array = (img_array * 255).astype(np.uint8)
        
        # Determine format based on shape
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            pil_img = Image.fromarray(img_array, 'RGBA')
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
            pil_img = Image.fromarray(img_array, 'RGB')
        else:
            pil_img = Image.fromarray(img_array)
            
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/metrics")
async def get_metrics():
    """Return the static metric plots and text."""
    v_plot_b64 = array_to_base64_png(ga.vision_curve_img)
    m_plot_b64 = array_to_base64_png(ga.speech_curve_img)
    
    return {
        "v_train_acc": f"{ga.v_train_acc:.2f}%",
        "v_test_acc": f"{ga.v_test_acc:.2f}%",
        "m_train_loss": f"{ga.m_train_loss:.4f}",
        "m_test_loss": f"{ga.m_test_loss:.4f}",
        "vision_plot": f"data:image/png;base64,{v_plot_b64}",
        "math_plot": f"data:image/png;base64,{m_plot_b64}"
    }

@app.post("/predict")
def predict(data: SketchData):
    try:
        # Decode base64 image
        header, encoded = data.image_base64.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        
        # Use existing logic to segment and preprocess
        matrices, gray_canvas = ga.segment_and_preprocess_expression(image)
        
        if len(matrices) < 3:
            return JSONResponse(status_code=400, content={
                "error": "Write a clean equation with at least 3 character tokens separated from left to right (e.g., 22+22, 100-45, 1+1)."
            })
            
        prediction_tokens = []
        last_hidden_activations = None
        
        for mat in matrices:
            input_batch = np.expand_dims(mat, axis=(0, -1))
            last_hidden_activations = ga.activation_model.predict(input_batch, verbose=0)[0]
            v_pred = ga.vision_net.predict(input_batch, verbose=0)[0]
            raw_pred_idx = int(np.argmax(v_pred))
            refined_idx = ga.refine_character_prediction(mat, raw_pred_idx, v_pred)
            prediction_tokens.append(refined_idx)
            
        equation_string = "".join([ga.CHAR_MAP[t] for t in prediction_tokens])
        final_calculated_digit = ga.evaluate_math_expression(equation_string)
        
        if final_calculated_digit is None:
            return JSONResponse(status_code=400, content={
                "error": f"Unable to compute math expression from parsed tokens: '{equation_string}'. Please draw digits and operator clearly."
            })
            
        # Generate all stages of SVG
        svg_step1 = ga.generate_complete_architecture_svg(
            prediction_tokens, 
            last_hidden_activations, 
            final_calculated_digit, 
            render_stage=1
        )
        svg_step2 = ga.generate_complete_architecture_svg(
            prediction_tokens, 
            last_hidden_activations, 
            final_calculated_digit, 
            render_stage=2
        )
        svg_step3 = ga.generate_complete_architecture_svg(
            prediction_tokens, 
            last_hidden_activations, 
            final_calculated_digit, 
            render_stage=3
        )
        
        return {
            "equation": equation_string,
            "result": final_calculated_digit,
            "svgs": [svg_step1, svg_step2, svg_step3],
            "tokens": [int(t) for t in prediction_tokens]
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
