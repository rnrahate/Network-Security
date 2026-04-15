import os
import sys

from network_security.components.data_ingestion import DataIngestion
from network_security.components.data_validation import DataValidation
from network_security.components.data_transformation import DataTransformation
from network_security.components.model_trainer import ModelTrainer
from network_security.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    try:
        if not os.getenv("DAGSHUB_USER_TOKEN"):
            raise EnvironmentError(
                "Missing DAGSHUB_USER_TOKEN. Configure your DagsHub access token "
                "in the environment before running the training pipeline."
            )

        # Initialize the training pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()

        # Initialize and run data ingestion
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info(f"Starting Data Ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed successfully")
        print(f"Data Ingestion Artifact: {data_ingestion_artifact}")
        logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")

        # Initialize and run data validation
        logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact = data_validation.initiate_validation()
        logging.info(f"Data Validation completed successfully")
        print(f"Data Validation Artifact: {data_validation_artifact}")
        logging.info(f"Data Validation Artifact: {data_validation_artifact}")

        # Initialize and run data transformation
        logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config, data_validation_artifact=data_validation_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(f"Data Transformation completed successfully")
        print(f"Data Transformation Artifact: {data_transformation_artifact}")
        logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")

        # Initialize and run model trainer and model evaluation
        logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info(f"Model Trainer completed successfully")
        print(f"Model Trainer Artifact: {model_trainer_artifact}")
        logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
