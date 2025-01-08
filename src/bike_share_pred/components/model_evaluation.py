import pandas as pd
import numpy as np
import os 
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.bike_share_pred import logger
import joblib

class ModelEval():
    def __init__(self,model,X_test,y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_model(self):
        try:
            self.y_pred = self.model.predict(self.X_test)
            self.mse = mean_squared_error(self.y_test,self.y_pred)
            self.mae = mean_absolute_error(self.y_test,self.y_pred)
            self.r2 = r2_score(self.y_test,self.y_pred)
            return self.mse,self.mae,self.r2
        except Exception as e:
            raise e

    def save_model(self):
        try:
            joblib.dump(self.model,os.path.join("models","model.pkl"))
        except Exception as e:
            raise e

    def save_metrics(self):
        try:
            with open("metrics.txt","w") as f:
                f.write(f"MSE: {self.mse}\n")
                f.write(f"MAE: {self.mae}\n")
                f.write(f"R2: {self.r2}\n")
        except Exception as e:
            raise e

