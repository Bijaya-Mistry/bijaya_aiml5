import joblib
from app.utils import preprocess_image

def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def predict_mask(image_bytes: bytes, model):
    img = preprocess_image(image_bytes)
    prediction = model.predict(img)
    return "Mask" if prediction[0] > 0.5 else "No Mask"
