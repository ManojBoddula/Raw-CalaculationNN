import tensorflow as tf

print("Loading vision_model.keras...")
vision_model = tf.keras.models.load_model('vision_model.keras', compile=False)
print("Saving as vision_model.h5...")
vision_model.save('vision_model.h5')

print("Loading math_model.keras...")
math_model = tf.keras.models.load_model('math_model.keras', compile=False)
print("Saving as math_model.h5...")
math_model.save('math_model.h5')

print("Conversion successful!")
