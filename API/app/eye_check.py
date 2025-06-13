import cv2
import tempfile

def is_eye_image(image_bytes):
    """
    Check if the uploaded image contains an eye using OpenCV's Haar Cascade.

    Args:
        image_bytes (bytes): The image file in bytes format.

    Returns:
        bool: True if an eye is detected, False otherwise.
    """
    # Save the image temporarily to load it with OpenCV
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file.write(image_bytes)
        tmp_path = tmp_file.name

    image = cv2.imread(tmp_path)
    if image is None:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    return len(eyes) > 0
