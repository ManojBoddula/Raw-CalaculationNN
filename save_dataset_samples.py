import os
import numpy as np
from PIL import Image
from vision_nn import generate_organic_char, CHAR_MAP

output_dir = "dataset_samples"
os.makedirs(output_dir, exist_ok=True)

print(f" Generating sample images of the trained dataset in '{output_dir}'...")

for char in CHAR_MAP:
    char_label = f"class_{char}" if char.isdigit() else f"class_op_{CHAR_MAP.index(char)}"
    char_dir = os.path.join(output_dir, char_label)
    os.makedirs(char_dir, exist_ok=True)
    
    for i in range(5):
        img_array = generate_organic_char(char)
        img_255 = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_255).resize((140, 140), Image.Resampling.NEAREST)
        pil_img.save(os.path.join(char_dir, f"sample_{i+1}.png"))

print(f" Successfully generated dataset PNG samples in: {os.path.abspath(output_dir)}")
