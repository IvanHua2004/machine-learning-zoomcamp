"""Serve the model as a web service with FastAPI.

Run locally:
    pipenv run uvicorn predict:app --host 0.0.0.0 --port 9696
Docs (interactive): http://localhost:9696/docs
"""

import pickle
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from features import add_features

MODEL_PATH = "model.bin"
# TODO(threshold): pick this from the precision/recall analysis in the notebook
THRESHOLD = 0.5

with open(MODEL_PATH, "rb") as f_in:
    pipeline = pickle.load(f_in)

app = FastAPI(title="Machine Failure Prediction")


class MachineReading(BaseModel):
    type: Literal["L", "M", "H"] = Field(description="Product quality variant")
    air_temperature: float = Field(gt=0, description="Kelvin")
    process_temperature: float = Field(gt=0, description="Kelvin")
    rotational_speed: float = Field(gt=0, description="rpm")
    torque: float = Field(ge=0, description="Nm")
    tool_wear: float = Field(ge=0, description="minutes")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "L",
                    "air_temperature": 298.9,
                    "process_temperature": 309.1,
                    "rotational_speed": 1350,
                    "torque": 62.5,
                    "tool_wear": 220,
                }
            ]
        }
    }


class Prediction(BaseModel):
    failure_probability: float
    failure: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(reading: MachineReading) -> Prediction:
    df = add_features(pd.DataFrame([reading.model_dump()]))
    prob = float(pipeline.predict_proba(df.to_dict(orient="records"))[0, 1])
    return Prediction(failure_probability=prob, failure=prob >= THRESHOLD)
