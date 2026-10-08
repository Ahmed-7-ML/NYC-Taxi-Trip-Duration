""" """

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
from .schemas import TripRequest, PredictionResponse
from .model_loader import load_model, get_State

# Model must be loaded only once <-- IMPORTANT
@asynccontextmanager
async def lifespan(app: FastAPI):
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
    state = get_State()
    if state.model is None:
        raise HTTPException(
            status_code=404,
            detail='Model Not Loaded!!'
        )
    
    df = pd.DataFrame([trip_request.model_dump()])
    X = state.preprocessor.transform(df)
    duration = float(state.model.predict(X)[0])
    
    return PredictionResponse(
        prediction_duration_minutes=round(duration, 2),
        model_version=state.version,
        model_alias=state.alias
    )

# -------------------------------------------------------

# uv run -m uvicorn main:app --reload
# uv run python -m uvicorn main:app --reload
# --> Run a Python Module
# يشغّل uvicorn كـ Python module (زي python -m uvicorn)

# uv run uvicorn main:app
# --> بيشغّل الـ executable بتاع uvicorn (اللي موجود في .venv/bin)

# الاتنين عادة بيشتغلوا كويس، لكن استخدام -m أحيانًا أفضل لو فيه مشاكل في الـ PATH أو في الـ entry points.
