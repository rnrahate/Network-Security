from sklearn.model_selection import GridSearchCV
import yaml
import os, sys
import numpy as np
import dill 
import pickle
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.utils.ml_utils.metric.classification_metric import get_classification_score

def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def write_yaml_file(file_path:str, content:object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as yaml_file:
            yaml.dump(content, yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def save_numpy_array_data(file_path:str, array:np.array) -> None: #type: ignore
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def save_pickle_object(file_path:str, obj:object) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_pickle_object(file_path:str) -> object:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Pickle file not found at path: {file_path}")
        with open(file_path, "rb") as file_obj:
            print(f"Loading pickle object from {file_path}")
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_numpy_array_data(file_path:str) -> np.array: #type: ignore
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def evaluate_models(X_train, y_train, X_test, y_test, models:dict, param:dict):
    try:
        report = {}
        for model_name, model in models.items():
            para = param[model_name]
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = get_classification_score(y_train, y_train_pred)
            test_model_score = get_classification_score(y_test, y_test_pred)

            report[model_name] = test_model_score.accuracy_score
        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e