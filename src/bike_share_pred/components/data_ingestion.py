import os
import sys
import pandas as pd

# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.bike_share_pred import logger

class DataIngestion():
    def __init__(self):
        self.logger = logger
        self.logger.info("DataIngestion object created")
    
    def read_data(self):
        try:
            file_path = os.path.join('artifacts', 'hour.csv')
            self.logger.info(f"Reading data from {file_path}")
            self.data = pd.read_csv(file_path)
            self.logger.info("Data read successfully")
            return self.data
        except Exception as e:
            self.logger.error(f"Error occurred while reading data: {str(e)}")
            raise e

if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        data = data_ingestion.read_data()
        print(data.head())
    except Exception as e:
        print(f"Error: {str(e)}")