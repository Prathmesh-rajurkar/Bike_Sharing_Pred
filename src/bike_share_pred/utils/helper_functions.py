import joblib
import os
from src.bike_share_pred import logger

def load_model(model_path: str):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e