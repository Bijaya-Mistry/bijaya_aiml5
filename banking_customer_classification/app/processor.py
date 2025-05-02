import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
from typing import Dict, Any, List, Union
import logging
import warnings
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BankingCustomerClassifier:
    """
    A classifier for identifying potential long-term investors among banking customers
    using XGBoost with hyperparameter tuning.
    """
    
    def __init__(self, model_dir="models"):
        """
        Initialize the classifier.
        
        Args:
            model_dir: Directory where models will be saved
        """
        self.model = None
        self.preprocessor = None
        self.feature_importance = None
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "banking_investor_model.pkl")
        self.preprocessor_path = os.path.join(model_dir, "banking_preprocessor.pkl")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
    def preprocess_data(self, 
                      data: pd.DataFrame, 
                      numeric_features: List[str], 
                      categorical_features: List[str]) -> ColumnTransformer:
        """
        Creates a preprocessing pipeline for banking customer data.
        
        Args:
            data: DataFrame containing the banking data
            numeric_features: List of numerical feature column names
            categorical_features: List of categorical feature column names
            
        Returns:
            A fitted column transformer for preprocessing the data
        """
        # Create preprocessing pipelines for both numeric and categorical data
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    
    def train(self, 
             data: pd.DataFrame, 
             target_column: str,
             numeric_features: List[str],
             categorical_features: List[str],
             test_size: float = 0.2,
             random_state: int = 42,
             param_grid: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Trains an XGBoost classifier with hyperparameter tuning to identify potential
        long-term investors.
        
        Args:
            data: DataFrame containing the banking data
            target_column: Name of the column containing the target variable
            numeric_features: List of numerical feature column names
            categorical_features: List of categorical feature column names
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            param_grid: Dictionary of parameters for XGBoost hyperparameter tuning
                        If None, a default grid will be used
        
        Returns:
            Dictionary containing model evaluation metrics and other information
        """
        logger.info("Starting model training process")
        
        # Split the data into features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        # Create and fit the preprocessor
        self.preprocessor = self.preprocess_data(X_train, numeric_features, categorical_features)
        
        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
                'xgbclassifier__max_depth': [3, 5, 7],
                'xgbclassifier__n_estimators': [100, 200],
                'xgbclassifier__subsample': [0.8, 1.0],
                'xgbclassifier__colsample_bytree': [0.8, 1.0],
                'xgbclassifier__gamma': [0, 0.1],
                'xgbclassifier__min_child_weight': [1, 3]
            }
        
        # Create a pipeline with preprocessing and XGBoost classifier
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('xgbclassifier', xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                use_label_encoder=False,
                random_state=random_state
            ))
        ])
        
        # Perform grid search with cross-validation
        logger.info("Starting hyperparameter tuning with GridSearchCV")
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the grid search
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search.fit(X_train, y_train)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        
        # Make predictions on the test set
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        # Calculate evaluation metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Extract feature importances if possible
        try:
            xgb_model = self.model.named_steps['xgbclassifier']
            feature_names = (
                numeric_features + 
                list(self.model.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features))
            )
            
            importances = xgb_model.feature_importances_
            self.feature_importance = dict(zip(feature_names, importances))
            
            # Sort feature importances
            self.feature_importance = {k: v for k, v in sorted(
                self.feature_importance.items(), 
                key=lambda item: item[1], 
                reverse=True
            )}
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {e}")
            self.feature_importance = {}
        
        # Save the model and preprocessor
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.preprocessor, self.preprocessor_path)
        
        logger.info(f"Model training completed. Best parameters: {grid_search.best_params_}")
        
        # Return training results
        results = {
            "model_accuracy": class_report['accuracy'],
            "model_precision": class_report['1']['precision'],  # For the positive class
            "model_recall": class_report['1']['recall'],        # For the positive class
            "model_f1": class_report['1']['f1-score'],          # For the positive class
            "roc_auc": roc_auc,
            "best_parameters": grid_search.best_params_,
            "cv_results_summary": {
                "mean_test_score": np.mean(grid_search.cv_results_['mean_test_score']),
                "std_test_score": np.mean(grid_search.cv_results_['std_test_score'])
            },
            "top_features": dict(list(self.feature_importance.items())[:10]) if self.feature_importance else {}
        }
        
        return results
    
    def predict(self, customer_data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Predicts whether customers are potential long-term investors.
        
        Args:
            customer_data: DataFrame or list of dictionaries containing customer information
            
        Returns:
            Dictionary with prediction results including probabilities and classification
        """
        if self.model is None:
            try:
                self.load_model()
            except Exception as e:
                raise Exception(f"Model has not been trained or saved. Error: {str(e)}")
        
        # Convert list of dictionaries to DataFrame if necessary
        if isinstance(customer_data, list):
            customer_data = pd.DataFrame(customer_data)
        
        # Make predictions
        probabilities = self.model.predict_proba(customer_data)[:, 1]
        predictions = self.model.predict(customer_data)
        
        # Format results
        results = []
        for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
            result = {
                "customer_index": i,
                "probability": float(prob),
                "is_potential_investor": bool(pred),
                "confidence": "high" if abs(prob - 0.5) > 0.3 else "medium" if abs(prob - 0.5) > 0.15 else "low"
            }
            results.append(result)
        
        return {
            "predictions": results,
            "prediction_summary": {
                "total_customers": len(customer_data),
                "potential_investors": int(sum(predictions)),
                "average_probability": float(np.mean(probabilities))
            }
        }
    
    def load_model(self) -> bool:
        """
        Loads a saved model and preprocessor.
        
        Returns:
            Boolean indicating whether the model was successfully loaded
        """
        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            
            # Try to extract feature importances
            try:
                xgb_model = self.model.named_steps['xgbclassifier']
                self.feature_importance = dict(zip(
                    self.model.named_steps['preprocessor'].get_feature_names_out(),
                    xgb_model.feature_importances_
                ))
                
                # Sort feature importances
                self.feature_importance = {k: v for k, v in sorted(
                    self.feature_importance.items(), 
                    key=lambda item: item[1], 
                    reverse=True
                )}
            except Exception as e:
                logger.warning(f"Could not extract feature importances during loading: {e}")
                self.feature_importance = {}
                
            return True
        except FileNotFoundError:
            logger.error("Model or preprocessor file not found")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False  
