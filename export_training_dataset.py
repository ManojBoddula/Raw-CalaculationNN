import os
import numpy as np
from PIL import Image
import tensorflow as tf
from vision_nn import generate_organic_char, CHAR_MAP

output_folder = "TRAINING_DATASET_IMAGES"
os.makedirs(output_folder, exist_ok=True)


# 1. Export Operator/Digit stroke samples
for idx, char_symbol in enumerate(CHAR_MAP):
    safe_name = char_symbol if char_symbol.isalnum() else f"op_{CHAR_MAP.index(char_symbol)}"
    class_folder = os.path.join(output_folder, f"Class_{idx}_{safe_name}")
    os.makedirs(class_folder, exist_ok=True)
    
    for sample_num in range(15):
        img_mat = generate_organic_char(char_symbol)
        img_255 = (img_mat * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_255).resize((140, 140), Image.Resampling.NEAREST)
        pil_img.save(os.path.join(class_folder, f"synthetic_sample_{sample_num+1}.png"))

# 2. Export MNIST digit samples
(x_train_raw, y_train_raw), _ = tf.keras.datasets.mnist.load_data()
for digit in range(10):
    class_folder = os.path.join(output_folder, f"Class_{digit}_{digit}")
    matching_indices = np.where(y_train_raw == digit)[0][:15]
    for sample_num, sample_idx in enumerate(matching_indices):
        pil_img = Image.fromarray(x_train_raw[sample_idx]).resize((140, 140), Image.Resampling.NEAREST)
        pil_img.save(os.path.join(class_folder, f"mnist_sample_{sample_num+1}.png"))

print("DONE! All training dataset images saved ")
