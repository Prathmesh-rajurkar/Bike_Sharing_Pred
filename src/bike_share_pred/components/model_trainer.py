import os 
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.bike_share_pred import logger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

class ModelTrainer():
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.logger = logger
        self.logger.info("ModelTrainer object created")

    def train_model(self):
        try:
            self.logger.info("Training the model")
            self.model = LinearRegression()
            self.model.fit(self.X_train,self.y_train)
            self.logger.info("Model trained successfully")
            return self.model
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {str(e)}")
            raise e

