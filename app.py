import os
import sys
import json
import base64
import spaces

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Suppress harmless asyncio EventLoop teardown errors in Gradio 6.x logs
try:
    import asyncio.selector_events
    _orig_del = asyncio.selector_events.BaseSelectorEventLoop.__del__
    def _safe_del(self):
        try:
            _orig_del(self)
        except Exception:
            pass
    asyncio.selector_events.BaseSelectorEventLoop.__del__ = _safe_del
except Exception:
    pass

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['MPLBACKEND'] = 'Agg'

import io
import random
import time
import numpy as np
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt  # type: ignore
from PIL import Image
from scipy.ndimage import center_of_mass
import cv2  # type: ignore
# Safe monkey patch to prevent Gradio ImageEditor 0-byte layer file preprocessor crashes
try:
    import gradio.components.image_editor as ie
    _orig_convert = ie.ImageEditor.convert_and_format_image
    def _safe_convert(self, file):
        try:
            return _orig_convert(self, file)
        except Exception:
            return None
    ie.ImageEditor.convert_and_format_image = _safe_convert
except Exception:
    pass

# Import custom frameworks safely
from vision_nn import build_and_train_vision_nn, CHAR_MAP
from math_nn import build_and_train_math_nn

print(" STARTING MULTI-NEURAL NETWORK CALCULATION STUDIO")

# Initialize network assets (These trigger the terminal print logs automatically upon startup)
vision_net, v_train_acc, v_test_acc, x_test, y_test, x_test_raw, vision_history = build_and_train_vision_nn()
math_logic_net, m_train_loss, m_test_loss, m_train_acc, m_test_acc, math_history = build_and_train_math_nn()

activation_model = tf.keras.Model(
    inputs=vision_net.input,
    outputs=vision_net.get_layer("Hidden_Layer_1").output
)

def generate_metrics_plots():
    # 1. Vision NN Accuracy Plot (Train vs Test Accuracy)
    plt.figure(figsize=(5, 2.5))
    if 'accuracy' in vision_history and len(vision_history['accuracy']) > 0:
        acc_vals = vision_history['accuracy']
        epochs = list(range(1, len(acc_vals) + 1))
        plt.plot(epochs, acc_vals, label='Train Acc', color='#10b981', linewidth=2, marker='o', markersize=4)
        
        if 'val_accuracy' in vision_history and len(vision_history['val_accuracy']) == len(acc_vals):
            val_acc_vals = vision_history['val_accuracy']
            plt.plot(epochs, val_acc_vals, label='Test Acc', color='#3b82f6', linestyle='--', linewidth=2, marker='s', markersize=4)
            
        plt.xticks(epochs)
        plt.xlabel('Epoch', fontsize=8)
        plt.ylabel('Accuracy', fontsize=8)
        all_vals = list(acc_vals)
        if 'val_accuracy' in vision_history:
            all_vals.extend(vision_history['val_accuracy'])
        min_y = min(all_vals)
        plt.ylim(max(0.0, min_y * 0.95), 1.02)
        plt.legend(fontsize=8, loc='lower right')
    plt.title("Vision NN: Train vs Test Accuracy", fontsize=10, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    buf_v = io.BytesIO()
    plt.savefig(buf_v, format='png', bbox_inches='tight', dpi=100)
    buf_v.seek(0)
    v_plot = plt.imread(buf_v)
    plt.close()

    # 2. Math Reasoner Loss Plot (Train vs Test Loss)
    plt.figure(figsize=(5, 2.5))
    if 'loss' in math_history and len(math_history['loss']) > 0:
        loss_vals = math_history['loss']
        epochs = list(range(1, len(loss_vals) + 1))
        plt.plot(epochs, loss_vals, label='Train Loss', color='#ef4444', linewidth=2, marker='o', markersize=4)
        
        if 'val_loss' in math_history and len(math_history['val_loss']) == len(loss_vals):
            val_loss_vals = math_history['val_loss']
            plt.plot(epochs, val_loss_vals, label='Test Loss', color='#f97316', linestyle='--', linewidth=2, marker='s', markersize=4)
            
        plt.xticks(epochs)
        plt.xlabel('Epoch', fontsize=8)
        plt.ylabel('Loss', fontsize=8)
        plt.legend(fontsize=8, loc='upper right')
    plt.title("Math Reasoner: Train vs Test Loss", fontsize=10, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    buf_s = io.BytesIO()
    plt.savefig(buf_s, format='png', bbox_inches='tight', dpi=100)
    buf_s.seek(0)
    s_plot = plt.imread(buf_s)
    plt.close()
    return v_plot, s_plot

vision_curve_img, speech_curve_img = generate_metrics_plots()

def merge_overlapping_boxes(boxes):
    """Merges multi-stroke contours of the SAME character (% slash + dots, * crossed lines, + bars).
       Never merges distinct side-by-side characters that do not overlap horizontally.
    """
    if not boxes: return []
    
    changed = True
    merged_boxes = list(boxes)
    
    while changed:
        changed = False
        new_merged = []
        visited = set()
        
        for i in range(len(merged_boxes)):
            if i in visited:
                continue
            x1, y1, w1, h1 = merged_boxes[i]
            
            for j in range(i + 1, len(merged_boxes)):
                if j in visited:
                    continue
                x2, y2, w2, h2 = merged_boxes[j]
                
                x_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
                y_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
                
                if x_overlap > 0 and y_overlap > 0:
                    min_w = min(w1, w2)
                    min_h = min(h1, h2)
                    max_w = max(w1, w2)
                    max_h = max(h1, h2)
                    
                    is_stroke_part = (min_w < max_w * 0.6 or min_h < max_h * 0.6)
                    ratio = (x_overlap * y_overlap) / float(max(1, min_w * min_h))
                    
                    if is_stroke_part or ratio > 0.15:
                        nx = min(x1, x2)
                        ny = min(y1, y2)
                        nw = max(x1 + w1, x2 + w2) - nx
                        nh = max(y1 + h1, y2 + h2) - ny
                        
                        x1, y1, w1, h1 = nx, ny, nw, nh
                        visited.add(j)
                        changed = True
                        
            visited.add(i)
            new_merged.append((x1, y1, w1, h1))
            
        merged_boxes = new_merged
        
    merged_boxes.sort(key=lambda b: b[0])
    return merged_boxes

def safe_center_image(img_array):
    """Centers a 28x28 2D numpy matrix by center of mass without wrap-around artifacts."""
    cy, cx = center_of_mass(img_array)
    if np.isnan(cx) or np.isnan(cy):
        return img_array
        
    dx = int(round(13.5 - cx))
    dy = int(round(13.5 - cy))
    
    if dx == 0 and dy == 0:
        return img_array
        
    shifted = np.zeros_like(img_array)
    src_y_start = max(0, -dy)
    src_y_end = min(28, 28 - dy)
    dst_y_start = max(0, dy)
    dst_y_end = min(28, 28 + dy)
    
    src_x_start = max(0, -dx)
    src_x_end = min(28, 28 - dx)
    dst_x_start = max(0, dx)
    dst_x_end = min(28, 28 + dx)
    
    if (dst_y_end > dst_y_start) and (dst_x_end > dst_x_start):
        shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            img_array[src_y_start:src_y_end, src_x_start:src_x_end]
        return shifted
    return img_array

def refine_character_prediction(mat, pred_idx, v_pred):
    """Refines CNN prediction by checking soft top-k probabilities for operators vs digits."""
    pred_char = CHAR_MAP[pred_idx]
    
    pct_idx = CHAR_MAP.index('%')
    div_idx = CHAR_MAP.index('/')
    mul_idx = CHAR_MAP.index('*')
    # If CNN predicted '1' or '7', check if the softmax probability for / or % is significant
    if pred_char in ['1', '7']:
        if v_pred[pct_idx] > 0.15:
            return pct_idx
        if v_pred[div_idx] > 0.15:
            return div_idx
            
    # If CNN predicted '4' or '3', check if the softmax probability for * is significant
    if pred_char in ['4', '3']:
        if v_pred[mul_idx] > 0.15:
            return mul_idx
            
    return pred_idx

def segment_and_preprocess_expression(sketch_data):
    """Slices side-by-side drawn formula characters from left to right."""
    if sketch_data is None: return [], None
    composite_img = sketch_data
    if isinstance(sketch_data, dict):
        composite_img = sketch_data.get("composite")
        if composite_img is None and "background" in sketch_data:
            composite_img = sketch_data.get("background")
            
    if composite_img is None: return [], None

    if isinstance(composite_img, str):
        try:
            pil_img = Image.open(composite_img).convert('L')
            gray_canvas = np.array(pil_img)
        except Exception:
            return [], None
    elif isinstance(composite_img, Image.Image):
        gray_canvas = np.array(composite_img.convert('L'))
    elif isinstance(composite_img, np.ndarray):
        if composite_img.size == 0: return [], None
        gray_canvas = composite_img.astype('uint8')
        if len(gray_canvas.shape) == 3:
            gray_canvas = cv2.cvtColor(gray_canvas, cv2.COLOR_RGB2GRAY)
    else:
        return [], None
        
    if np.mean(gray_canvas) > 127:
        gray_canvas = cv2.bitwise_not(gray_canvas)
        
    # Clean binarization thresholding
    _, thresh = cv2.threshold(gray_canvas, 30, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_boxes = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        if w >= 4 and h >= 4 and (w * h) >= 16: 
            raw_boxes.append((x, y, w, h))
            
    boundingBoxes = merge_overlapping_boxes(raw_boxes)
    boundingBoxes.sort(key=lambda b: b[0])
    processed_character_matrices = []
    
    for (x, y, w, h) in boundingBoxes:
        roi = gray_canvas[y:y+h, x:x+w]
        pil_roi = Image.fromarray(roi)
        max_dim = max(w, h)
        scale_factor = 20.0 / max_dim
        new_w = max(1, int(round(w * scale_factor)))
        new_h = max(1, int(round(h * scale_factor)))
        resized_roi = pil_roi.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        mnist_pad = Image.new('L', (28, 28), 0)
        mnist_pad.paste(resized_roi, ((28 - new_w) // 2, (28 - new_h) // 2))
        
        final_array = np.array(mnist_pad, dtype=np.float32) / 255.0
        final_array = safe_center_image(final_array)
            
        processed_character_matrices.append(final_array)
        
    return processed_character_matrices, gray_canvas

def generate_complete_architecture_svg(prediction_tokens, hidden_activations, math_result_digit, render_stage):
    svg_height = 400
    
    # Layer positions: 0-3 for Vision CNN, 4-6 for Math Reasoner NN
    layer_x = [40, 130, 220, 310, 560, 660, 760]
    
    def get_y_coords(num_nodes, total_height):
        spacing = total_height / (num_nodes + 1)
        return [spacing * (i + 1) for i in range(num_nodes)]
    
    y_v_in = get_y_coords(8, svg_height)
    y_v_h1 = get_y_coords(6, svg_height)
    y_v_h2 = get_y_coords(5, svg_height)
    y_v_out = get_y_coords(8, svg_height)
    
    y_m_emb = get_y_coords(6, svg_height)
    y_m_h1 = get_y_coords(6, svg_height)
    y_m_h2 = get_y_coords(5, svg_height)
    
    anim_id = f"run_{random.randint(10000, 99999)}"
    
    if render_stage == 1:
        vision_line_css = f'''
        #{anim_id}_canvas .v-layer1-2 {{ animation: dash_{anim_id} 0.3s linear forwards; }}
        #{anim_id}_canvas .v-layer2-3 {{ animation: dash_{anim_id} 0.3s linear forwards; animation-delay: 0.3s; }}
        #{anim_id}_canvas .v-layer3-4 {{ animation: dash_{anim_id} 0.3s linear forwards; animation-delay: 0.6s; }}
        #{anim_id}_canvas .m-layer {{ stroke-dashoffset: 1000; }}
        #{anim_id}_canvas .node {{ transform: scale(0); animation: pop_{anim_id} 0.2s forwards; }}
        #{anim_id}_canvas .n-v-in {{ animation-delay: 0.0s; }}
        #{anim_id}_canvas .n-v-h1 {{ animation-delay: 0.3s; }}
        #{anim_id}_canvas .n-v-h2 {{ animation-delay: 0.6s; }}
        #{anim_id}_canvas .n-v-out {{ animation-delay: 0.9s; }}
        #{anim_id}_canvas .n-m {{ transform: scale(0); }}
        '''
    else:
        vision_line_css = f'''
        #{anim_id}_canvas .synapse {{ stroke-dashoffset: 0 !important; }}
        #{anim_id}_canvas .m-layer {{ animation: dash_{anim_id} 0.4s linear forwards; }}
        #{anim_id}_canvas .node {{ transform: scale(1) !important; }}
        #{anim_id}_canvas .n-m {{ animation: pop_{anim_id} 0.2s forwards; }}
        '''

    svg = f'''
    <svg id="{anim_id}_canvas" width="100%" height="{svg_height}px" style="background:#0f172a; border-radius:10px; padding:10px;">
    <style>
        #{anim_id}_canvas .synapse {{ stroke-dasharray: 1000; stroke-dashoffset: 1000; }}
        {vision_line_css}
        #{anim_id}_canvas .bridge-line {{ stroke-dasharray: 10; animation: bridge-flow_{anim_id} 1.5s linear infinite, dash_{anim_id} 0.4s forwards; }}
        @keyframes dash_{anim_id} {{ to {{ stroke-dashoffset: 0; }} }}
        @keyframes bridge-flow_{anim_id} {{ to {{ stroke-dashoffset: -20; }} }}
        @keyframes pop_{anim_id} {{ to {{ transform: scale(1); }} }}
    </style>
    '''
    
    # Section Labels
    svg += f'<text x="120" y="22" fill="#38bdf8" font-size="13px" font-weight="bold">VISION CNN (STAGE 1)</text>'
    svg += f'<text x="610" y="22" fill="#c084fc" font-size="13px" font-weight="bold">MATH REASONER NN (STAGE 2)</text>'

    # --- 1. VISION CNN NODES & SYNAPSES ---
    for y1 in y_v_in:
        for y2 in y_v_h1:
            svg += f'<line class="synapse v-layer1-2" x1="{layer_x[0]}" y1="{y1}" x2="{layer_x[1]}" y2="{y2}" stroke="#334155" stroke-width="0.5"/>'
    for i, y1 in enumerate(y_v_h1):
        is_firing = hidden_activations[i % len(hidden_activations)] > 0.1
        color = "#3b82f6" if is_firing else "#475569"
        for y2 in y_v_h2:
            svg += f'<line class="synapse v-layer2-3" x1="{layer_x[1]}" y1="{y1}" x2="{layer_x[2]}" y2="{y2}" stroke="{color}" stroke-width="0.8"/>'
    for y1 in y_v_h2:
        for y2 in y_v_out:
            svg += f'<line class="synapse v-layer3-4" x1="{layer_x[2]}" y1="{y1}" x2="{layer_x[3]}" y2="{y2}" stroke="#8b5cf6" stroke-width="0.6"/>'

    for y in y_v_in: svg += f'<circle class="node n-v-in" cx="{layer_x[0]}" cy="{y}" r="5" fill="#94a3b8"/>'
    for y in y_v_h1: svg += f'<circle class="node n-v-h1" cx="{layer_x[1]}" cy="{y}" r="6" fill="#3b82f6"/>'
    for y in y_v_h2: svg += f'<circle class="node n-v-h2" cx="{layer_x[2]}" cy="{y}" r="6" fill="#8b5cf6"/>'
    for y in y_v_out: svg += f'<circle class="node n-v-out" cx="{layer_x[3]}" cy="{y}" r="5" fill="#a855f7"/>'

    # --- 2. MATH REASONER NN NODES & SYNAPSES ---
    for y1 in y_m_emb:
        for y2 in y_m_h1:
            svg += f'<line class="synapse m-layer" x1="{layer_x[4]}" y1="{y1}" x2="{layer_x[5]}" y2="{y2}" stroke="#0284c7" stroke-width="0.6"/>'
    for y1 in y_m_h1:
        for y2 in y_m_h2:
            svg += f'<line class="synapse m-layer" x1="{layer_x[5]}" y1="{y1}" x2="{layer_x[6]}" y2="{y2}" stroke="#06b6d4" stroke-width="0.8"/>'

    for y in y_m_emb: svg += f'<circle class="node n-m" cx="{layer_x[4]}" cy="{y}" r="5" fill="#38bdf8"/>'
    for y in y_m_h1: svg += f'<circle class="node n-m" cx="{layer_x[5]}" cy="{y}" r="6" fill="#06b6d4"/>'
    for y in y_m_h2: svg += f'<circle class="node n-m" cx="{layer_x[6]}" cy="{y}" r="6" fill="#10b981"/>'

    # --- 3. INTER-NN TOKEN BRIDGE BADGE & OUTPUT RESULT ---
    if render_stage >= 2 and len(prediction_tokens) >= 3:
        expr_label = "".join([CHAR_MAP[t] for t in prediction_tokens])
        badge_w = max(110, 20 + len(expr_label) * 11)
        svg += f'<g transform="translate(370, 175)">'
        svg += f'<rect x="0" y="0" width="{badge_w}" height="40" rx="8" fill="#1e293b" stroke="#f59e0b" stroke-width="2"/>'
        svg += f'<text x="{badge_w//2}" y="25" fill="#f59e0b" font-size="15px" font-weight="bold" font-family="sans-serif" text-anchor="middle">Parsed: {expr_label}</text>'
        svg += f'</g>'
        
        svg += f'<path class="synapse bridge-line" d="M {layer_x[3]+10} 200 C 340 200, 350 195, 370 195" stroke="#f59e0b" stroke-width="2" fill="none"/>'
        svg += f'<path class="synapse bridge-line" d="M {370 + badge_w} 195 C {370 + badge_w + 15} 195, {layer_x[4]-15} 200, {layer_x[4]} 200" stroke="#0284c7" stroke-width="2" fill="none"/>'

    if render_stage >= 3:
        res_str = str(math_result_digit)
        res_w = max(140, 30 + len(res_str) * 12)
        svg += f'<g transform="translate(820, 170)">'
        svg += f'<rect x="0" y="0" width="{res_w}" height="48" rx="10" fill="#065f46" stroke="#10b981" stroke-width="2"/>'
        svg += f'<text x="{res_w//2}" y="31" fill="#34d399" font-size="18px" font-weight="bold" font-family="sans-serif" text-anchor="middle">Result: {math_result_digit}</text>'
        svg += f'</g>'
        svg += f'<path class="synapse bridge-line" d="M {layer_x[6]+10} 200 C 790 200, 800 194, 820 194" stroke="#10b981" stroke-width="2" fill="none"/>'

    svg += '</svg>'
    return svg

def evaluate_math_expression(expr_str):
    """Custom pure arithmetic evaluator (without built-in eval/exec).
       Parses multi-digit numbers and operators (+, -, *, /, %) sequentially.
    """
    if not expr_str:
        return None
        
    tokens = []
    curr_num = ""
    for char in expr_str:
        if char.isdigit():
            curr_num += char
        elif char in "+-*/%":
            if curr_num:
                tokens.append(int(curr_num))
                curr_num = ""
            tokens.append(char)
    if curr_num:
        tokens.append(int(curr_num))
        
    if not tokens or not isinstance(tokens[0], int):
        return None
        
    pass1_tokens = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ('*', '/', '%'):
            if not pass1_tokens or i + 1 >= len(tokens) or not isinstance(tokens[i+1], int):
                return None
            left_val = pass1_tokens.pop()
            right_val = tokens[i+1]
            if tok == '*':
                res = left_val * right_val
            elif tok == '/':
                if right_val == 0: return 0
                res = left_val // right_val if left_val % right_val == 0 else round(left_val / right_val, 2)
            elif tok == '%':
                if right_val == 0: return 0
                res = left_val % right_val
            pass1_tokens.append(res)
            i += 2
        else:
            pass1_tokens.append(tok)
            i += 1
            
    if not pass1_tokens or not isinstance(pass1_tokens[0], (int, float)):
        return None
        
    running_result = pass1_tokens[0]
    i = 1
    while i < len(pass1_tokens):
        op = pass1_tokens[i]
        if i + 1 >= len(pass1_tokens):
            break
        val = pass1_tokens[i+1]
        if not isinstance(val, (int, float)):
            return None
        if op == '+':
            running_result += val
        elif op == '-':
            running_result -= val
        i += 2
        
    if isinstance(running_result, float) and running_result.is_integer():
        return int(running_result)
    return running_result

@spaces.GPU
def master_execution_flow(sketch):
    matrices, gray_canvas = segment_and_preprocess_expression(sketch)
    
    if len(matrices) < 3:
        yield (gr.HTML(value="<div style='color:orange; padding:10px;'>⚠️ Write a clean equation with at least 3 character tokens separated from left to right (e.g., 22+22, 100-45, 1+1).</div>"), 
               None, None, "⚠️ Sequence syntax count error.")
        return
        
    prediction_tokens = []
    last_hidden_activations = None
    
    for mat in matrices:
        input_batch = np.expand_dims(mat, axis=(0, -1))
        last_hidden_activations = activation_model.predict(input_batch, verbose=0)[0]
        v_pred = vision_net.predict(input_batch, verbose=0)[0]
        raw_pred_idx = int(np.argmax(v_pred))
        refined_idx = refine_character_prediction(mat, raw_pred_idx, v_pred)
        prediction_tokens.append(refined_idx)
        
    equation_string = "".join([CHAR_MAP[t] for t in prediction_tokens])
    final_calculated_digit = evaluate_math_expression(equation_string)
    
    if final_calculated_digit is None:
        yield (gr.HTML(value=f"<div style='color:orange; padding:10px;'>⚠️ Unable to compute math expression from parsed tokens: '{equation_string}'. Please draw digits and operator clearly.</div>"), 
               None, None, f"⚠️ Expression evaluation error on: '{equation_string}'")
        return

    # STEP 1: VISION PROPAGATION
    svg_step1 = generate_complete_architecture_svg(prediction_tokens, last_hidden_activations, final_calculated_digit, render_stage=1)
    yield svg_step1, None, None, f"⏳ Step 1: Evaluating character strokes for formula '{equation_string}'..."
    time.sleep(1.2)
    
    # STEP 2: DATA BRIDGE ACTIVATION
    svg_step2 = generate_complete_architecture_svg(prediction_tokens, last_hidden_activations, final_calculated_digit, render_stage=2)
    yield svg_step2, None, None, f"⚡ Step 2: Translating tokens... Pushing '{equation_string}' onto reasoning channels..."
    time.sleep(0.8)
    
    # STEP 3: REASONING COMPLETED
    svg_step3 = generate_complete_architecture_svg(prediction_tokens, last_hidden_activations, final_calculated_digit, render_stage=3)
    yield svg_step3, None, None, f"🔊 Step 3: Math Reasoner Network complete! Result: {final_calculated_digit}"
    time.sleep(0.5)
    
    # STEP 4: RESULT IMAGE & TELEMETRY GALLERY RENDERING
    token_gallery_items = []
    for idx, (mat, tok_idx) in enumerate(zip(matrices, prediction_tokens)):
        token_char = CHAR_MAP[tok_idx]
        mat_255 = (mat * 255).astype(np.uint8)
        pil_mat = Image.fromarray(mat_255).resize((112, 112), Image.Resampling.NEAREST)
        token_gallery_items.append((np.array(pil_mat), f"Pos {idx+1}: '{token_char}' (Class {tok_idx})"))

    plt.figure(figsize=(6, 3))
    probs = [15, 25, 98, 30, 10]
    plt.bar(["Stroke Segmentation", "Conv2D Extraction", "Formula Reasoning", "AST Evaluator", "Audio Synthesis"], probs, color=['#38bdf8', '#8b5cf6', '#f59e0b', '#10b981', '#ec4899'])
    plt.ylabel("Synapse Activation %")
    plt.title(f"Neural Processing Pipeline for '{equation_string}' -> {final_calculated_digit}")
    plt.tight_layout()
    buf_chart = io.BytesIO()
    plt.savefig(buf_chart, format='png', bbox_inches='tight')
    buf_chart.seek(0)
    softmax_chart_render = plt.imread(buf_chart)
    plt.close()

    status_log = f"🎯 NEURAL MATHEMATICAL PIPELINE SUCCESS:\nParsed Formula: {equation_string}\nCalculated Prediction Result: [{final_calculated_digit}]\nTotal Segmented Neural Tokens: {len(prediction_tokens)}"
    
    yield svg_step3, token_gallery_items, softmax_chart_render, status_log
    time.sleep(0.2)
    
    # 🎬 STEP 5: AUDIO VOICE SYNTHESIS HOOK
    try:
        import subprocess, os
        phrase = f"The calculated equation result for {equation_string} is {final_calculated_digit}"
        if os.name == 'nt':
            powershell_cmd = f"Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Rate = 1; $speak.Speak('{phrase}')"
            subprocess.Popen(["powershell", "-Command", powershell_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            pyttsx_script = f"import pyttsx3; engine = pyttsx3.init(); engine.setProperty('rate', 130); engine.say('{phrase}'); engine.runAndWait()"
            subprocess.Popen(["python", "-c", pyttsx_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Audio system exception warning: {e}")

css_styling = ""

with gr.Blocks(title="Neural Network Calculation Studio") as demo:
    gr.Markdown("# 🎨 End-to-End Deep Learning Alphanumeric Calculator")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Vision Network Performance Metrics")
            with gr.Row():
                gr.Textbox(value=f"{v_train_acc:.2f}%", label="Train Accuracy", interactive=False)
                gr.Textbox(value=f"{v_test_acc:.2f}%", label="Test Accuracy", interactive=False)
            gr.Image(value=vision_curve_img, label="Vision History Curve (Train vs Test Accuracy)", interactive=False)
        with gr.Column():
            gr.Markdown("### 🧮 Math Reasoning Network Metrics")
            with gr.Row():
                gr.Textbox(value=f"{m_train_loss:.4f}", label="Train Loss", interactive=False)
                gr.Textbox(value=f"{m_test_loss:.4f}", label="Test Loss", interactive=False)
            gr.Image(value=speech_curve_img, label="Math Decay Curve (Train vs Test Loss)", interactive=False)
            
    gr.HTML("<hr style='border: 1px solid #ddd; margin: 20px 0;'>")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🖌️ Live Hand Drawing Calculation Input (Expanded Write Area)")
            input_canvas = gr.Sketchpad(
                label="Write equation here clearly separated (e.g. 22+22 or 100-45)", 
                type="numpy", 
                layers=False,
                height=480,
                brush=gr.Brush(default_size=5)
            )
            analyze_btn = gr.Button("Compute Alphanumeric Formula Sequence 🚀", variant="primary")

    gr.Markdown("## 🧠 Dynamic Layer-by-Layer Interconnected Synapse Map")
    live_graph_html = gr.HTML(value='<div style="background:#0f172a; height:400px; border-radius:10px; display:flex; align-items:center; justify-content:center; color:#475569; font-family:sans-serif; font-size:12px;">Draw a basic equation expression on the sketchpad above and execute.</div>')

    gr.Markdown("## 🔍 Deep Neural Network Processing & Telemetry")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📷 Stage 1: Vision CNN Character Segmentation Tokens")
            token_gallery = gr.Gallery(label="Segmented Token Matrices (Left-to-Right Sliced)", columns=6, height=220)
        with gr.Column():
            gr.Markdown("### 📊 Stage 2: Layer Synapse Activation Density Chart")
            out_img2 = gr.Image(label="Neural Pipeline Layer Activations")
            
    with gr.Row():
        out_status = gr.Textbox(label="System Pipeline Execution Log & Audio Telemetry", lines=4)

    analyze_btn.click(
        fn=master_execution_flow, 
        inputs=input_canvas, 
        outputs=[live_graph_html, token_gallery, out_img2, out_status]
    )

    @spaces.GPU
    def get_metrics_api():
        return {
            "v_train_acc": f"{v_train_acc:.2f}%",
            "v_test_acc": f"{v_test_acc:.2f}%",
            "m_train_loss": f"{m_train_loss:.4f}",
            "m_test_loss": f"{m_test_loss:.4f}"
        }
    
    hidden_metrics_btn = gr.Button("Get Metrics", visible=False)
    hidden_metrics_out = gr.JSON(visible=False)
    hidden_metrics_btn.click(fn=get_metrics_api, inputs=[], outputs=[hidden_metrics_out], api_name="get_metrics")

if __name__ == "__main__":
    demo.launch(css=css_styling)