import sys,os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from network_security.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS,TARGET_COLUMN
from network_security.entity.config_entity import DataTransformationConfig
from network_security.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.utils.main_utils.utils import save_numpy_array_data, save_pickle_object

class DataTransformation:
    def __init__(self, data_transformation_config:DataTransformationConfig, data_validation_artifact:DataValidationArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def read_data(file_path:str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def get_data_transformer_object(self) -> Pipeline:
        try:
            logging.info(f"Creating data transformer object")
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            pipeline = Pipeline(steps=[
                ("imputer", imputer)
            ])
            logging.info(f"Data transformer object created successfully")
            return pipeline
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info(f"Entering the data transformation method")
        try:
            logging.info(f"Initiating data transformation")
            # read train and test data
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)
            logging.info(f"Read train and test data for data transformation")   

            # separate input features and target feature from train and test dataframe and replace target feature value with 0 and 1
            X_train = train_df.drop(TARGET_COLUMN, axis=1)
            y_train = train_df[TARGET_COLUMN].replace({-1: 0, 1: 1})
            X_test = test_df.drop(TARGET_COLUMN, axis=1)
            y_test = test_df[TARGET_COLUMN].replace({-1: 0, 1: 1})
            logging.info(f"Separated input features and target feature from train and test dataframe")

            # get data transformer object
            preprocessor = self.get_data_transformer_object()

            # fit and transform the train and test data
            preprocessor_object = preprocessor.fit(X_train)
            X_train_transformed = preprocessor_object.transform(X_train)
            X_test_transformed = preprocessor_object.transform(X_test)
            logging.info(f"Transformed train and test data using the data transformer object")

            # train and test array after transformation
            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]
            logging.info(f"Combined transformed input features and target feature into train and test array")

            # save transformed train and test array and preprocessor object
            transformed_train_file_path = self.data_transformation_config.transformed_train_file_path
            transformed_test_file_path = self.data_transformation_config.transformed_test_file_path
            preprocessor_object_file_path = self.data_transformation_config.preprocessor_object_file_path

            save_numpy_array_data(file_path=transformed_train_file_path, array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path, array=test_arr)
            save_pickle_object(file_path=preprocessor_object_file_path, obj=preprocessor_object)
            logging.info(f"Saved transformed train and test array and preprocessor object")

            final_model_dir = os.path.join(os.getcwd(), "final_model")
            save_pickle_object(os.path.join(final_model_dir, "preprocessor.pkl"), preprocessor_object)
            logging.info(f"Saved preprocessing object to final model folder: {final_model_dir}")

            # prepare artifact
            data_transformation_artifact = DataTransformationArtifact(
                    transformed_train_file_path=transformed_train_file_path,
                    transformed_test_file_path=transformed_test_file_path,
                    preprocessor_object_file_path=preprocessor_object_file_path
                )
            return data_transformation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e