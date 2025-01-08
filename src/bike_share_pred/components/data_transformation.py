import os 
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.bike_share_pred import logger

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


if __name__ == "__main__":
    try:
        data_transformation = DataTransformation(data)
        data = data_transformation.drop_columns()
        print(data.head())
    except Exception as e:
        print(f"Error: {str(e)}")