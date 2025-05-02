import cv2
import numpy as np

def preprocess_image(image_bytes: bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0  # Normalize
        img = img.reshape(1, 224, 224, 3)
        return img
    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {e}")
 
