MLFlow 

	- uv run mlflow ui --backend-store-uri sqlite:///mlflow_trip_duration.db

Prefect

	- uv run prefect server start

	- uv run python main.py --sample-size [n] --no-tune/--tune --promote/-

FastAPI

	- uv run -m uvicorn src.api.main:app
