# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import mlflow

app = FastAPI()

# Carrega o modelo na inicialização
model = joblib.load("model.joblib")

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(features: IrisFeatures):
    mlflow.start_run(nested=True)
    data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]
    prediction = model.predict(data)[0]
    mlflow.log_metric("predicted_class", int(prediction))
    mlflow.end_run()
    return {"predicted_class": int(prediction)}
