import os 
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.bike_share_pred import logger
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.model_selection import train_test_split
file_path = os.path.join('artifacts', 'hour.csv')
data=pd.read_csv(file_path)

class DataTransformation():
    def __init__(self,data):
        self.data = data
        self.logger = logger
        self.logger.info("DataTransformation object created")

    def drop_columns(self):
        try:
            self.data.drop(columns=['weathersit','temp','atemp','hum','windspeed','instant','dteday'],inplace=True)
            return self.data
        except Exception as error:
            logger.error("Error occurred while dropping columns",error)
            raise Exception("Error in drop_columns method",error)

    def train_test_split(self):
        try:
            self.logger.info("Splitting data into train and test sets")
            self.X = self.data.drop(columns=['cnt'])
            self.y = self.data['cnt']
            X_scaled = scaler.fit_transform(self.X)
            y_scaled = scaler.fit_transform(self.y.values.reshape(-1,1))
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X_scaled,y_scaled,test_size=0.2,random_state=42)
            self.logger.info("Data split successfully")
            return self.X_train,self.X_test,self.y_train,self.y_test
        except Exception as e:
            self.logger.error(f"Error occurred while splitting data: {str(e)}")
            raise e


if __name__ == "__main__":
    try:
        data_transformation = DataTransformation(data)
        data = data_transformation.drop_columns()
        print(data.head())
        X_train,X_test,y_train,y_test = data_transformation.train_test_split()
        print(X_train[0],y_train[0])
        print(X_train.shape,y_train.shape)
    except Exception as e:
        print(f"Error: {str(e)}")