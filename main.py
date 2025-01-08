from src.bike_share_pred.components.data_ingestion import DataIngestion
from src.bike_share_pred.components.data_transformation import DataTransformation
from src.bike_share_pred.components.model_trainer import ModelTrainer
from src.bike_share_pred.components.model_evaluation import ModelEval
from src.bike_share_pred import logger

STAGE_NAME = "Data Ingestion"

try:
    logger.info(f"\n>>>>> Stage : {STAGE_NAME} started <<<<<<<")
    data_ingestion = DataIngestion()
    data = data_ingestion.read_data()
    logger.info(f">>>>> Stage : {STAGE_NAME} ended <<<<<<<\n")
except Exception as error:
    logger.error(f"Error occurred: {str(error)}")
    raise error


STAGE_NAME = "Data Transformation"

try:
    logger.info(f"\n>>>>> Stage : {STAGE_NAME} started <<<<<<<")
    data_transformation = DataTransformation(data)
    data = data_transformation.drop_columns()
    print(data.head())
    X_train,X_test,y_train,y_test = data_transformation.train_test_split()
    print(X_train[0],y_train[0])
    print(X_train.shape,y_train.shape)
    logger.info(f">>>>> Stage : {STAGE_NAME} ended <<<<<<<\n")
except Exception as error:
    logger.error(f"Error occurred: {str(error)}")
    raise error


STAGE_NAME = "Model Training"

try:
    logger.info(f"\n>>>>> Stage : {STAGE_NAME} started <<<<<<<")
    model_trainer = ModelTrainer(X_train,X_test,y_train,y_test)
    model = model_trainer.train_model()
    logger.info(f">>>>> Stage : {STAGE_NAME} ended <<<<<<<\n")
except Exception as error:
    logger.error(f"Error occurred: {str(error)}")
    raise error

STAGE_NAME = "Model Evaluation"

try:
    logger.info(f"\n>>>>> Stage : {STAGE_NAME} started <<<<<<<")
    model_eval = ModelEval(model,X_test,y_test)
    mse,mae,r2 = model_eval.evaluate_model()
    model_eval.save_model()
    model_eval.save_metrics()
    logger.info(f">>>>> Stage : {STAGE_NAME} ended <<<<<<<\n")
except Exception as error:
    logger.error(f"Error occurred: {str(error)}")
    raise error