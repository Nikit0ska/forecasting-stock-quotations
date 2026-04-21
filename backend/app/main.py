from datetime import timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from app.data import load_series, ASSETS
from app.preprocessing import train_test_split
from app.metrics import regression_metrics
from app.schemas import ForecastResponse

from app.models import naive, moving_average, arima, exponential_smoothing

app = FastAPI(title="Stock Forecasting System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/assets")
def assets():
    return ASSETS

@app.get("/forecast", response_model=ForecastResponse)
def forecast(asset: str, model: str = "naive", steps: int = 20):
    series = load_series(asset, start_date = "2025-10-01")
    train, test = train_test_split(series)

    if model == "naive":
        pred = naive.forecast(train, len(test) + steps)
    elif model == "ma":
        pred = moving_average.forecast(train, len(test)+ steps)
    elif model == "arima":
        pred = arima.forecast(train, len(test)+ steps)
    elif model == "exp":
        pred = exponential_smoothing.forecast(train, len(test)+ steps)
    else:
        raise ValueError("Unknown model")


    y_true = np.array(test, dtype=float)
    y_pred = np.array(pred, dtype=float)
    metrics = regression_metrics(y_true, y_pred[:len(y_true)])
    dates = train.keys().tolist()

    current_date = dates[-1]
    for i in range(len(test) + steps):
        dates.append(current_date)
        current_date += timedelta(days=1)


    return {
        "train": train.tolist(),
        "test": test.tolist(),
        "forecast": pred,
        "metrics": metrics,
        "dates": [i.strftime("%Y-%m-%d") for i in dates],
    }
