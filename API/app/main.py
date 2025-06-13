from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
from eye_check import is_eye_image
from model_utils import load_model, load_encoder, save_encoder
from preprocess import preprocess_image_from_upload
from predict_utils import make_prediction
from train_utils import fine_tune_model
import pandas as pd
from reset import reset_project


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI(title="Iris Eye Recognition API")

cwd = os.getcwd()
USER_DATA_PATH = os.path.join(cwd, "df_info.csv")
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
        df = pd.read_csv(USER_DATA_PATH)
        print(predicted_class)
        matched_rows = df[df["ID_Number"].astype(str) == str(predicted_class)]

        if matched_rows.empty:
            return JSONResponse(
                content={"error": f"No user found with ID {predicted_class}"},
                status_code=404
            )

        record = matched_rows.to_dict(orient="records")[0]
        return JSONResponse(content=record)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/add_data_and_train")
async def add_data_and_train(
    file: UploadFile = File(...),
    name: str = Form(...),
    address: str = Form(...),
    id_number: str = Form(...),
    birth_date: str = Form(...),
    reset_all_app: bool = Form(False)
):
    global model, encoder

    try:
        contents = await file.read()

        if not is_eye_image(contents):
            return JSONResponse(
                content={"error": "The uploaded image does not appear to contain a clear eye region."},
                status_code=400
            )

        input_image = preprocess_image_from_upload(contents)

        if reset_all_app:
            reset_project()

        label = id_number
        message = fine_tune_model(model, input_image, label, name, address, birth_date)
        

        return JSONResponse(content={"message": message})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
