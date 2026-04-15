import os
import sys
import pandas as pd
import numpy as np
"""
Defining common constants for training pipeline
"""
TARGET_COLUMN = "Result"
PIPELINE_NAME = str("network_security")
ARTIFACT_DIR = os.path.join(os.getcwd(), "artifacts")
FILE_NAME = str("phishingData.csv")
TRAIN_FILE_NAME = str("train.csv")
TEST_FILE_NAME = str("test.csv")
SAVED_MODEL_DIR = os.path.join("saved_models")

"""
Data ingestion related start with DATA_INGESTION VAR_NAME
"""
DATA_INGESTION_COLLECTION_NAME = str("traffic_data")
DATA_INGESTION_DATABASE_NAME = str("network_traffic")
DATA_INGESTION_DIR_NAME = str("data_ingestion")
DATA_INGESTION_FEATURE_STORE_DIR = str("feature_store")
DATA_INGESTION_INGESTED_DIR = str("ingested")
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = float(0.2)

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

"""
Data Validation related start with DATA_VALIDATION VAR_NAME
"""
DATA_VALIDATION_DIR_NAME: str="data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation related start with DATA_TRANSFORMATION VAR_NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing_object.pkl"
## KNN Imputer params for data transformation
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}

"""
Model trainer related start with MODEL_TRAINER VAR_NAME
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_FILE_NAME: str = "best_model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVERFIITING_UNDERFITTING_THRESHOLD: float = 0.05