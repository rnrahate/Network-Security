from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

## configuration of the data_ingestion config
from network_security.entity.config_entity import DataIngestionConfig
from network_security.constants import training_pipeline
from network_security.entity.artifact_entity import DataIngestionArtifact
import os 
import sys
import pymongo
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL") or os.getenv("MongoDB_URL")
if not MONGO_DB_URL:
    raise EnvironmentError("MONGO_DB_URL is not set. Define it in .env or as an environment variable.")

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    def export_collection_as_dataframe(self, collection_name:str, database_name:str) -> pd.DataFrame:
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            with pymongo.MongoClient(MONGO_DB_URL, serverSelectionTimeoutMS=5000) as client:
                client.server_info() # to check if connection is successful or not
                db = client[database_name]
                collection = db[collection_name]
                data = list(collection.find())
            df = pd.DataFrame(data)
            logging.info(f"Exported Collection Data shape: {df.shape}")
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def split_data_as_train_test(self, dataframe:pd.DataFrame) -> dict:
        try:
            logging.info(f"Splitting data into train and test set")
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42)
            logging.info(f"Split data into train and test set with ratio {self.data_ingestion_config.train_test_split_ratio}")
            # save train and test set in ingested dir
            ingested_dir = self.data_ingestion_config.ingested_dir
            os.makedirs(ingested_dir, exist_ok=True)
            train_file_path = os.path.join(ingested_dir, training_pipeline.TRAIN_FILE_NAME)
            test_file_path = os.path.join(ingested_dir, training_pipeline.TEST_FILE_NAME)
            train_set.to_csv(train_file_path, index=False)
            test_set.to_csv(test_file_path, index=False)
            logging.info(f"Saved train and test set in ingested dir at {ingested_dir}")
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
        return {
                "train_path": train_file_path,
                "test_path": test_file_path
            }

    def initiate_data_ingestion(self):
        try:
            logging.info(f"Initiating data ingestion")
            df = self.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name, database_name=self.data_ingestion_config.database_name)
            # check if dataframe is empty
            if df.empty:
                raise ValueError("No data found in MongoDB collection")
            # drop _id column if exists
            if "_id" in df.columns:
                df.drop("_id", axis=1, inplace=True)
            # replace na with NAN
            df.replace(to_replace="na", value=pd.NA, inplace=True)
            # save data in feature store
            feature_store_dir = self.data_ingestion_config.feature_store_dir
            os.makedirs(feature_store_dir, exist_ok=True)
            feature_store_file_path = os.path.join(feature_store_dir, training_pipeline.FILE_NAME)
            df.to_csv(feature_store_file_path, index=False)
            logging.info(f"Saved data in feature store at {feature_store_file_path}")
            # split data into train and test set
            split_data_paths = self.split_data_as_train_test(dataframe=df)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
        data_ingestion_artifact = DataIngestionArtifact(feature_store_path=feature_store_file_path, train_path=split_data_paths["train_path"], test_path=split_data_paths["test_path"])
        logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")

        return data_ingestion_artifact

    
    