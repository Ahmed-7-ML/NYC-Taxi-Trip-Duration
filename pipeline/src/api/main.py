""" """

from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd
from schemas import TripRequest, PredictionResponse
from model_loader import load_model

# Model must be loaded only once <-- IMPORTANT
@asynccontextmanager
def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="NYC Taxi Trip Duration API",
    description="Predict trip duration from pickup/dropoff zone IDs (2019 TLC model)",
    version="1.0.0",
    lifespan=lifespan
)

@app.get('/health')
def health():
    return {
        'status': 'healthy',
        'developer': 'Ahmed Akram Amer'
    }

@app.post('/predict', response_model=PredictionResponse)
def predict(trip_request: TripRequest):
    ...