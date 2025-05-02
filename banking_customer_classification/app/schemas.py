  
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class TrainingConfig(BaseModel):
    """Configuration for model training"""
    target_column: str
    numeric_features: List[str]
    categorical_features: List[str]
    test_size: Optional[float] = Field(default=0.2, ge=0.1, le=0.5)
    random_state: Optional[int] = 42
    param_grid: Optional[Dict[str, Any]] = None

class CustomerData(BaseModel):
    """Customer data for prediction"""
    customers: List[Dict[str, Any]]

class PredictionResult(BaseModel):
    """Result of a single customer prediction"""
    customer_index: int
    probability: float = Field(ge=0.0, le=1.0)
    is_potential_investor: bool
    confidence: str = Field(...)

class PredictionSummary(BaseModel):
    """Summary of all predictions"""
    total_customers: int
    potential_investors: int
    average_probability: float = Field(ge=0.0, le=1.0)

class PredictionDetail(BaseModel):
    """Detailed prediction results"""
    predictions: List[PredictionResult]
    prediction_summary: PredictionSummary

class PredictionResponse(BaseModel):
    """Response for prediction endpoints"""
    status: str
    predictions: PredictionDetail

class ModelMetrics(BaseModel):
    """Model performance metrics"""
    model_accuracy: float = Field(ge=0.0, le=1.0)
    model_precision: float = Field(ge=0.0, le=1.0)
    model_recall: float = Field(ge=0.0, le=1.0)
    model_f1: float = Field(ge=0.0, le=1.0)
    roc_auc: float = Field(ge=0.0, le=1.0)
    best_parameters: Dict[str, Any]
    cv_results_summary: Dict[str, float]
    top_features: Dict[str, float]

class ModelResponse(BaseModel):
    """Response for model training endpoint"""
    status: str
    message: str
    results: ModelMetrics

class ModelInfoResponse(BaseModel):
    """Response for model info endpoint"""
    status: str
    model_type: str = "XGBoost"
    parameters: Dict[str, Any]
    feature_importance: Union[Dict[str, float], str]