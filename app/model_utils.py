import tensorflow as tf
import joblib
import os

cwd = os.getcwd()

MODEL_PATH = os.path.join(cwd, "IRISRecognizer.h5")
ENCODER_PATH = os.path.join(cwd, "encoder.joblib")

def load_model(path=MODEL_PATH):
    try:
        model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def save_model(model, path=MODEL_PATH):
    model.save(path)
    print(f"Model saved to {path}")

def load_encoder(path=ENCODER_PATH):
    if os.path.exists(path):
        encoder = joblib.load(path)
        return encoder
    else:
        raise FileNotFoundError("Label encoder not found")

def save_encoder(encoder, path=ENCODER_PATH):
    joblib.dump(encoder, path)
    print(f"Label encoder saved to {path}")
