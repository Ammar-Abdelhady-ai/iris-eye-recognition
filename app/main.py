from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
from eye_check import is_eye_image
from model_utils import load_model, load_encoder, save_encoder
from preprocess import preprocess_image_from_upload
from predict_utils import make_prediction
from train_utils import fine_tune_model


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI(title="Iris Eye Recognition API")


model = load_model()
encoder = load_encoder()



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
        input_image = preprocess_image_from_upload(contents)
        predicted_class, confidence = make_prediction(model, input_image)
        print(predicted_class)
        encoder = load_encoder()

        predicted_class = encoder.inverse_transform([predicted_class])[0]
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/add_data_and_train")
async def add_data_and_train(file: UploadFile = File(...), label: str = Form(...)):
    global model, encoder

    try:
        contents = await file.read()

        if not is_eye_image(contents):
            return JSONResponse(
                content={"error": "The uploaded image does not appear to contain a clear eye region."},
                status_code=400
            )

        input_image = preprocess_image_from_upload(contents)

        message = fine_tune_model(model, input_image, label)

        return JSONResponse(content={"message": message})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
