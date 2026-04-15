import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
print(f"Mongo DB URL: {mongo_db_url}")

import pymongo
from network_security.pipeline.training_pipeline import TrainingPipeline
from network_security.entity.config_entity import TrainingPipelineConfig
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from fastapi.responses import Response
from uvicorn import run as app_run
from starlette.responses import RedirectResponse
from network_security.utils.main_utils.utils import load_pickle_object
from network_security.utils.ml_utils.model.estimator import NetworkModel
import pandas as pd

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from network_security.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME,DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origin = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/",tags=["Authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train",tags=["Training"])
async def train_route():
    try:
        training_pipeline_config = TrainingPipelineConfig()
        train_pipeline = TrainingPipeline(training_pipeline_config=training_pipeline_config)
        train_pipeline.run_pipeline()
        return Response(content="Training pipeline executed successfully", media_type="text/plain")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

@app.post("/predict", tags=["Prediction"])
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_pickle_object("final_model/preprocessor.pkl")
        final_model=load_pickle_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        print(df.iloc[0])
        
        # Drop the target column before prediction since the model was trained without it
        from network_security.constants.training_pipeline import TARGET_COLUMN
        if TARGET_COLUMN in df.columns:
            df = df.drop(TARGET_COLUMN, axis=1)
        
        y_pred = network_model.predict(df)
        print(y_pred)
        df["predicted_column"] = y_pred 
        print(df['predicted_column']) 
        df.to_csv("prediction_output/output.csv")
        table_html = df.to_html(classes="table table-striped")
        return templates.TemplateResponse(
            request=request,
            name="table.html",
            context={"table": table_html},
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
        
if __name__ == "__main__":
    try:
        app_run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
