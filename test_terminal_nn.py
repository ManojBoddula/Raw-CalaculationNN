import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import segment_and_preprocess_expression, refine_character_prediction, evaluate_math_expression, CHAR_MAP, vision_net

def draw_terminal_expression(expr_str):
    """Generates an 1100x480 canvas image matching Gradio Sketchpad input."""
    canvas = Image.new('L', (1100, 480), 0)
    draw = ImageDraw.Draw(canvas)
    
    font_size = 70
    font = None
    for fn in ["arial.ttf", "calibri.ttf", "times.ttf", "DejaVuSans.ttf"]:
        try:
            font = ImageFont.truetype(fn, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
        
    x_cursor = 100
    y_cursor = 180
    
    for char in expr_str:
        if char == ' ':
            x_cursor += 30
            continue
            
        draw_char = char
        if char == '*':
            draw_char = 'x'
            
        bbox = draw.textbbox((0, 0), draw_char, font=font)
        w = max(25, bbox[2] - bbox[0])
        draw.text((x_cursor, y_cursor), draw_char, fill=255, font=font)
        x_cursor += w + 60
        
    return np.array(canvas)

def run_terminal_test(expr_list):
    print("=" * 65)
    print(" 🧪 TERMINAL NEURAL NETWORK & MATH EVALUATOR SUITE")
    print("=" * 65)
    
    all_passed = True
    
    for idx, target_expr in enumerate(expr_list, 1):
        print(f"\n[Test {idx}] Drawing Expression: '{target_expr}'")
        canvas_img = draw_terminal_expression(target_expr)
        
        matrices, _ = segment_and_preprocess_expression(canvas_img)
        print(f" 🔹 Segmented Character Tokens: {len(matrices)}")
        
        prediction_tokens = []
        token_chars = []
        
        for mat in matrices:
            input_batch = np.expand_dims(mat, axis=(0, -1))
            v_pred = vision_net.predict(input_batch, verbose=0)[0]
            raw_pred_idx = int(np.argmax(v_pred))
            refined_idx = refine_character_prediction(mat, raw_pred_idx, v_pred)
            prediction_tokens.append(refined_idx)
            token_chars.append(CHAR_MAP[refined_idx])
            
        parsed_str = "".join(token_chars)
        calc_result = evaluate_math_expression(parsed_str)
        
        print(f" 🔹 Tokens Recognized : {token_chars}")
        print(f" 🔹 Parsed Formula    : '{parsed_str}'")
        print(f" 🔹 Calculated Result : {calc_result}")
        
        expected_parsed = target_expr.replace('x', '*').replace('X', '*').replace(' ', '')
        if parsed_str == expected_parsed:
            print(" ✅ PASSED: Clean Recognition!")
        else:
            print(f" ❌ MISMATCH: Expected '{expected_parsed}', Got '{parsed_str}'")
            all_passed = False
            
    print("\n" + "=" * 65)
    if all_passed:
        print(" 🎉 ALL TERMINAL TESTS PASSED PERFECTLY!")
    else:
        print(" ⚠️ SOME TESTS HAD TOKEN AMBIGUITIES.")
    print("=" * 65)

def interactive_terminal_mode():
    print("\n" + "=" * 65)
    print(" ⌨️  INTERACTIVE TERMINAL NEURAL NETWORK TESTER")
    print(" Type any formula (e.g., 50 / 2, 99 * 3, 10 % 3, 123 + 456)")
    print(" Type 'exit' or 'q' to quit.")
    print("=" * 65)
    
    while True:
        try:
            user_input = input("\nEnter math equation > ").strip()
            if not user_input or user_input.lower() in ['exit', 'q', 'quit']:
                print("Exiting interactive terminal test. Goodbye!")
                break
                
            canvas_img = draw_terminal_expression(user_input)
            matrices, _ = segment_and_preprocess_expression(canvas_img)
            
            prediction_tokens = []
            token_chars = []
            
            for mat in matrices:
                input_batch = np.expand_dims(mat, axis=(0, -1))
                v_pred = vision_net.predict(input_batch, verbose=0)[0]
                raw_pred_idx = int(np.argmax(v_pred))
                refined_idx = refine_character_prediction(mat, raw_pred_idx, v_pred)
                prediction_tokens.append(refined_idx)
                token_chars.append(CHAR_MAP[refined_idx])
                
            parsed_str = "".join(token_chars)
            calc_result = evaluate_math_expression(parsed_str)
            
            print(f"  👉 Segmented Tokens : {len(matrices)}")
            print(f"  👉 Recognized Tokens: {token_chars}")
            print(f"  👉 Parsed Expression: '{parsed_str}'")
            print(f"  🧠 Neural Result   : {calc_result}")
            
        except KeyboardInterrupt:
            print("\nExiting interactive terminal test. Goodbye!")
            break
        except Exception as e:
            print(f" ⚠️ Error evaluating input: {e}")

if __name__ == "__main__":
    test_cases = [
        "56 / 2",
        "96 % 3",
        "24 + 2",
        "222 + 236",
        "336 x 256",
        "76 % 2",
        "100 - 45"
    ]
    run_terminal_test(test_cases)
    
    # Launch interactive prompt mode
    interactive_terminal_mode()
