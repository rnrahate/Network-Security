import os,sys

from network_security.utils.main_utils.utils import load_numpy_array_data,load_pickle_object,save_pickle_object,evaluate_models
from network_security.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from network_security.entity.config_entity import ModelTrainerConfig
from network_security.utils.ml_utils.metric.classification_metric import get_classification_score
from network_security.utils.ml_utils.model.estimator import NetworkModel
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier
)
import mlflow
import mlflow.sklearn as mlflow_sklearn

class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    def track_mlflow(self, model, classification_train_metric, classification_test_metric):
        logging.info(f"Tracking model training metrics with MLflow")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Network_Security_Experiment")
        with mlflow.start_run():
            # Log training metrics
            mlflow.log_metric("train_f1_score", classification_train_metric.f1_score)
            mlflow.log_metric("train_precision_score", classification_train_metric.precision_score)
            mlflow.log_metric("train_recall_score", classification_train_metric.recall_score)
            mlflow.log_metric("train_accuracy_score", classification_train_metric.accuracy_score)
            
            # Log test metrics
            mlflow.log_metric("test_f1_score", classification_test_metric.f1_score)
            mlflow.log_metric("test_precision_score", classification_test_metric.precision_score)
            mlflow.log_metric("test_recall_score", classification_test_metric.recall_score)
            mlflow.log_metric("test_accuracy_score", classification_test_metric.accuracy_score)
            
            mlflow_sklearn.log_model(model, "best_model")
        logging.info(f"Model training metrics tracked with MLflow: test_accuracy={classification_test_metric.accuracy_score}")

    def train_model(self, X_train, y_train,X_test, y_test):
        models = {
            "Logistic Regression": LogisticRegression(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier()
        }

        params = {
            "Logistic Regression": {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"]
            },
            "K-Nearest Neighbors": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"]
            },
            "Decision Tree": {
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy", "log_loss"]
            },
            "Random Forest": {
                "n_estimators": [8,16,32,64,128,256],
                "max_depth": [3, 5, 7]
            },
            "Gradient Boosting": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.1, 0.5, .001],
                "subsample": [0.5, 0.8, 1.0, 0.01, 0.71, 0.28]
            },
            "AdaBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.1, 0.5, 1.0]
            }
        }
        model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                            models=models, param=params)
        # best model score from dict
        top_model_score = max(model_report.values())
        # get best model name from dict
        best_model_name = [model_name for model_name, model_score in model_report.items() if model_score == top_model_score][0]
        best_model = models[best_model_name]
        logging.info(f"Best model found on training and testing dataset is {best_model_name} with accuracy score: {top_model_score}")
        
        # predicting on train data
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_train, y_train_pred)
        logging.info(f"Classification metric on training data: {classification_train_metric}")

        # predicting on test data
        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_test, y_test_pred)
        logging.info(f"Classification metric on test data: {classification_test_metric}")
        
        # track the mlflow
        self.track_mlflow(best_model, classification_train_metric, classification_test_metric)

        preprocessor = load_pickle_object(file_path=self.data_transformation_artifact.preprocessor_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)
        logging.info(f"Saving best model at {model_dir_path}")

        Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_pickle_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model)
        logging.info(f"Best model saved at {self.model_trainer_config.trained_model_file_path} with model name: {best_model_name}")

        # model trainer artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
        )
        return model_trainer_artifact
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(f"Loading transformed training data")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_array = load_numpy_array_data(transformed_train_file_path)
            test_array = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            logging.info(f"Loaded transformed training data from {transformed_train_file_path} with shape {train_array.shape}")
            logging.info(f"Loaded transformed test data from {self.data_transformation_artifact.transformed_test_file_path} with shape {test_array.shape}")

            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]
            logging.info(f"Split training data into X_train with shape {X_train.shape} and y_train with shape {y_train.shape}")
            logging.info(f"Split test data into X_test with shape {X_test.shape} and y_test with shape {y_test.shape}")

            logging.info(f"Training the model")
            model_trainer_artifact = self.train_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            logging.info(f"Model training completed with model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
