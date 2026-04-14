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
