import numpy as np
from PIL import Image
import io

def preprocess_image_from_upload(image_bytes, target_size=(150, 150)):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, target_size[0], target_size[1], 1)
        return image_array
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")
