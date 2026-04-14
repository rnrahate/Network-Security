from network_security.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from network_security.entity.config_entity import DataValidationConfig
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.constants.training_pipeline import SCHEMA_FILE_PATH
from network_security.utils.main_utils.utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp
import pandas as pd
import os, sys

class DataValidation:
    def __init__(self, data_validation_config:DataValidationConfig, data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def read_data(file_path:str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def validate_number_of_columns(self, dataframe:pd.DataFrame) -> bool:
        try:
            schema_columns = [list(column.keys())[0] for column in self._schema_config["columns"]]
            expected_column_count = len(schema_columns)
            actual_column_count = dataframe.shape[1]
            if actual_column_count != expected_column_count:
                logging.error(
                    f"Expected {expected_column_count} columns, but found {actual_column_count} columns"
                )
                return False

            if list(dataframe.columns) != schema_columns:
                logging.error(
                    "Column names do not match the schema. "
                    f"Expected: {schema_columns}, found: {list(dataframe.columns)}"
                )
                return False
            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def detect_data_drift(self, base_df:pd.DataFrame, current_df:pd.DataFrame) -> bool:
        try:
            drift_report = {}
            numerical_columns = self._schema_config.get("numerical_columns", [])
            for column in numerical_columns:
                if column not in base_df.columns or column not in current_df.columns:
                    continue

                base_data = base_df[column]
                current_data = current_df[column]
                ks_result = ks_2samp(base_data, current_data)
                if hasattr(ks_result, "pvalue"):
                    p_value = ks_result.pvalue #type: ignore
                else:
                    _, p_value = ks_result
                drift_report[column] = float(p_value) #type: ignore

            os.makedirs(self.data_validation_config.drift_report_dir, exist_ok=True)
            write_yaml_file(self.data_validation_config.drift_report_file_path, drift_report)

            drift_status = False
            drifted_columns = [col for col, p in drift_report.items() if p < 0.05]

            drift_status = len(drifted_columns) > 0
            for column in drifted_columns:
                p_value = drift_report[column]
                logging.info(f"Data drift detected in column: {column} with p-value: {p_value}")
            
            return drift_status
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_validation(self) -> DataValidationArtifact:
        try:
            logging.info(f"Initiating data validation")
            train_file_path = self.data_ingestion_artifact.train_path
            test_file_path = self.data_ingestion_artifact.test_path
            train_df = self.read_data(train_file_path)
            test_df = self.read_data(test_file_path)

            if train_df.empty:
                raise ValueError("Train file is empty")
            logging.info(f"Train file is not empty with shape {train_df.shape}")

            if test_df.empty:
                raise ValueError("Test file is empty")
            logging.info(f"Test file is not empty with shape {test_df.shape}")

            status_train = self.validate_number_of_columns(train_df)
            status_test = self.validate_number_of_columns(test_df)
            if not status_train:
                raise ValueError("Train file does not have expected number of columns")
            logging.info(f"Train file has expected number of columns")
            if not status_test:
                raise ValueError("Test file does not have expected number of columns")
            logging.info(f"Test file has expected number of columns")

            status_drift = self.detect_data_drift(base_df=train_df, current_df=test_df)
            validation_status = not status_drift
            target_train_path = self.data_validation_config.valid_train_file_path if validation_status else self.data_validation_config.invalid_train_file_path
            target_test_path = self.data_validation_config.valid_test_file_path if validation_status else self.data_validation_config.invalid_test_file_path

            os.makedirs(os.path.dirname(target_train_path), exist_ok=True)
            os.makedirs(os.path.dirname(target_test_path), exist_ok=True)

            train_df.to_csv(target_train_path, index=False)
            test_df.to_csv(target_test_path, index=False)
            logging.info(f"Saved validated train file at {target_train_path}")
            logging.info(f"Saved validated test file at {target_test_path}")
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=target_train_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                valid_test_file_path=target_test_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_name=self.data_validation_config.drift_report_file_path,
            )
            
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
