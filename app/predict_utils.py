import numpy as np

def make_prediction(model, input_image):
    prediction = model.predict(input_image)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))
    return predicted_class, round(confidence, 3)
