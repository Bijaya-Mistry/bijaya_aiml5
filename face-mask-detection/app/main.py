from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.model import load_model, predict_mask
from app.schemas import PredictionResponse
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/mask_detector.model")

# Initialize FastAPI app
app = FastAPI(title="Face Mask Detection API")

# Load model once at startup
model = load_model(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Welcome to the Face Mask Detection API"}

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = predict_mask(contents, model)
        return {"prediction": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
