from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import os
from typing import Dict, Any

from app.processor import BankingCustomerClassifier
from app.schemas import TrainingConfig, CustomerData, ModelResponse, PredictionResponse
from app.utils import save_upload_file, get_file_path, validate_csv_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Banking Customer Investor Classification API",
    description="API for classifying banking customers as potential long-term investors",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create global classifier instance
classifier = BankingCustomerClassifier()

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint that returns API information"""
    return {
        "message": "Banking Customer Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/train": "Train the model using customer banking data",
            "/predict": "Predict whether customers are potential long-term investors",
            "/predict_file": "Predict using a CSV file containing customer data",
            "/model_info": "Get information about the trained model"
        }
    }

@app.post("/train", response_model=ModelResponse)
async def train_model(
    training_file: UploadFile = File(...), 
    config: str = None
):
    """
    Train the model using customer banking data.
    
    Args:
        training_file: CSV file containing customer banking data
        config: JSON string with training configuration
        
    Returns:
        Dictionary containing model evaluation metrics and training results
    """
    try:
        # Parse the configuration
        if config:
            config_dict = json.loads(config)
            training_config = TrainingConfig(**config_dict)
        else:
            raise HTTPException(status_code=400, detail="Training configuration is required")
        
        # Validate and save the CSV file
        file_path = await save_upload_file(training_file, "uploads")
        
        # Validate CSV file
        data = validate_csv_file(file_path)
        
        # Train the model
        results = classifier.train(
            data=data,
            target_column=training_config.target_column,
            numeric_features=training_config.numeric_features,
            categorical_features=training_config.categorical_features,
            test_size=training_config.test_size,
            random_state=training_config.random_state,
            param_grid=training_config.param_grid
        )
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(customer_data: CustomerData):
    """
    Predict whether customers are potential long-term investors.
    
    Args:
        customer_data: Customer data in JSON format
        
    Returns:
        Prediction results including probabilities and classifications
    """
    try:
        # Check if model is loaded, if not try to load it
        if classifier.model is None:
            if not classifier.load_model():
                raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")
        
        # Make predictions
        predictions = classifier.predict(customer_data.customers)
        
        return {
            "status": "success",
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_file", response_model=PredictionResponse)
async def predict_file(file: UploadFile = File(...)):
    """
    Predict using a CSV file containing customer data.
    
    Args:
        file: CSV file containing customer data
        
    Returns:
        Prediction results including probabilities and classifications
    """
    try:
        # Check if model is loaded, if not try to load it
        if classifier.model is None:
            if not classifier.load_model():
                raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")
        
        # Validate and save the CSV file
        file_path = await save_upload_file(file, "uploads")
        
        # Validate CSV file
        data = validate_csv_file(file_path)
        
        # Make predictions
        predictions = classifier.predict(data)
        
        return {
            "status": "success",
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model_info")
async def model_info():
    """
    Get information about the trained model.
    
    Returns:
        Dictionary containing model information
    """
    try:
        # Check if model is loaded, if not try to load it
        if classifier.model is None:
            if not classifier.load_model():
                raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")
        
        # Get model information
        model = classifier.model.named_steps.get('xgbclassifier')
        
        if model:
            return {
                "status": "success",
                "model_type": "XGBoost",
                "parameters": model.get_params(),
                "feature_importance": classifier.feature_importance if classifier.feature_importance else "Not available"
            }
        else:
            return {
                "status": "error",
                "message": "Model structure is not as expected"
            }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")  
