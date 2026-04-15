import os
import sys

from network_security.components.data_ingestion import DataIngestion
from network_security.components.data_validation import DataValidation
from network_security.components.data_transformation import DataTransformation
from network_security.components.model_trainer import ModelTrainer
from network_security.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from network_security.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

class TrainingPipeline:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            logging.info(f"{'>>'*20} Training Pipeline {'<<'*20}")
            self.training_pipeline_config = training_pipeline_config
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            self.data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            logging.info(f"Starting Data Ingestion")
            data_ingestion_artifact = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed successfully")
            print(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    def data_validation(self, data_ingestion_artifact:DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_validation()
            logging.info(f"Data Validation completed successfully")
            print(f"Data Validation Artifact: {data_validation_artifact}")
            logging.info(f"Data Validation Artifact: {data_validation_artifact}")

            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def data_transformation(self, data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_transformation_config=data_transformation_config, data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data Transformation completed successfully")
            print(f"Data Transformation Artifact: {data_transformation_artifact}")
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")

            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def start_model_trainer(self, data_transformation_artifact:DataTransformationArtifact):
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model Trainer completed successfully")
            print(f"Model Trainer Artifact: {model_trainer_artifact}")
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")

            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            logging.info(f"Training pipeline completed successfully")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e