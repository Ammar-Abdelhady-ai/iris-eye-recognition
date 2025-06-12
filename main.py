from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI(title="Iris Eye Recognition API")

try:
    model = tf.keras.models.load_model('model.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image_from_upload(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((150, 150))
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 150, 150, 1)
        return image_array
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

@app.get("/")
def read_root():
    return {
        "Project Name": "Iris Eye Recognition",
        "Description": "API for predicting iris class from eye images"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not model:
            return JSONResponse(content={"error": "Model is not loaded correctly."}, status_code=500)
        
        input_image = preprocess_image_from_upload(contents)
        prediction = model.predict(input_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        return JSONResponse(content={
            "predicted_class": int(predicted_class),
            "confidence": round(confidence, 3)
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

############### NEW ENDPOINT FOR FINE-TUNING ################
