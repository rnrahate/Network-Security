import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL = os.getenv("MongoDB_URL")
print(MONGO_DB_URL)

import certifi
ca = certifi.where()

import pandas as pd 
import numpy as np
import pymongo
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            logging.info("MongoDB client initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize MongoDB client: {str(e)}")
            raise NetworkSecurityException("Failed to initialize MongoDB client", sys.exc_info())
        
    def cv_to_json(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df.reset_index(drop=True, inplace=True)

            json_data = df.to_dict(orient='records')   # ✅ BEST FIX

            logging.info(f"CSV file at {file_path} converted to JSON successfully.")
            return json_data

        except Exception as e:
            logging.error(f"Failed to convert CSV to JSON: {str(e)}")
            raise NetworkSecurityException("Failed to convert CSV to JSON", sys.exc_info())
    def push_data_to_mongodb(self, json_data, database_name, collection_name):
        try:
            client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)  # ✅ fresh client

            db = client[database_name]
            collection = db[collection_name]

            collection.insert_many(json_data)   # ✅ direct use

            logging.info(f"Data pushed to MongoDB collection '{collection_name}' successfully.")

        except Exception as e:
            logging.error(f"Failed to push data to MongoDB: {str(e)}")
            raise NetworkSecurityException("Failed to push data to MongoDB", sys.exc_info())
if __name__ == "__main__":
    try:
        data_extractor = NetworkDataExtract()
        json_data = data_extractor.cv_to_json('C:\\Users\\Asus\\Documents\\Complete Data Science ML-DL-NLP Course\\PROJECTS\\Network-Security_Project_with_MLFlow_monitoring\\network_data\\phisingData.csv')
        data_extractor.push_data_to_mongodb(json_data, 'network_traffic', 'traffic_data')
    except NetworkSecurityException as e:
        logging.error(f"An error occurred: {str(e)}")