import tensorflow as tf
import numpy as np
import subprocess
import os
def build_and_train_speech_nn():
    
    # Generate an identity diagonal matrix representing 10 discrete categorical channels
    all_digits_one_hot = np.eye(10, dtype=np.float32)
    target_values = np.array([i for i in range(10)], dtype=np.int32)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),  
        tf.keras.layers.Dense(64, activation='relu', name="Speech_Input"),
        tf.keras.layers.Dense(64, activation='relu', name="Speech_Hidden"),
        tf.keras.layers.Dense(10, activation='softmax', name="Speech_Output_Classes")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # AGGRESSIVE EARLY STOPPING GUARD
    # Breaks execution if loss change is lower than 0.001 for 2 consecutive epochs
    speech_early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.001,         
        patience=2,                
        restore_best_weights=True
    )
    
    # Pass the callback into the fit parameters array
    history = model.fit(
        all_digits_one_hot, 
        target_values, 
        epochs=150,                
        callbacks=[speech_early_stop], 
        verbose=1
    )
    
    final_loss = history.history['loss'][-1]
    print(f"Speech Network Training Complete. Final Loss: {final_loss:.6f}")
    return model, final_loss, history.history

def speak_digit_live(digit_idx):
    digit_idx = int(digit_idx)
    
    digit_words = {
        0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
        5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
    }
    
    word_to_speak = digit_words.get(digit_idx, "Unknown Number")
    print(f"Audio Engine Speaking: '{word_to_speak}'")
    
    if os.name == 'nt':
        powershell_cmd = f"Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Rate = 1; $speak.Speak('{word_to_speak}')"
        try:
            subprocess.Popen(["powershell", "-Command", powershell_cmd],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return
        except Exception as e:
            print(f"PowerShell Speech Fallback: {e}")
            
    try:
        pyttsx_script = f"import pyttsx3; engine = pyttsx3.init(); engine.setProperty('rate', 125); engine.say('{word_to_speak}'); engine.runAndWait()"
        subprocess.Popen(["python", "-c", pyttsx_script],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Speech Engine Error: {e}")
