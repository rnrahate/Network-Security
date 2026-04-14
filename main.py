import sys
from network_security.components.data_validation import DataValidation
from network_security.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig
from network_security.components.data_ingestion import DataIngestion
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info(f"Starting Data Ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed successfully")
        print(f"Data Ingestion Artifact: {data_ingestion_artifact}")
        logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
        logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact = data_validation.initiate_validation()
        logging.info(f"Data Validation completed successfully")
        print(f"Data Validation Artifact: {data_validation_artifact}")
        logging.info(f"Data Validation Artifact: {data_validation_artifact}")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e