from datetime import timedelta, datetime

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
def forecast(asset: str, model: str = "naive", steps: int = 1, start_date: str = "2026-01-01"):
    series = load_series(asset, start_date = start_date)
    train, test = train_test_split(series)

    if model == "naive":
        val = naive.forecast(train, len(test))
        pred = naive.forecast(test, steps)
    elif model == "ma":
        val = moving_average.forecast(train, len(test))
        pred = moving_average.forecast(train, steps)
    elif model == "arima":
        val = arima.forecast(train, len(test))
        pred = arima.forecast(train, steps)
    elif model == "exp":
        val = exponential_smoothing.forecast(train, len(test))
        pred = exponential_smoothing.forecast(train, steps)
    else:
        raise ValueError("Unknown model")


    y_true = np.array(test, dtype=float)
    y_pred = np.array(val, dtype=float)
    metrics = regression_metrics(y_true, y_pred[:len(y_true)])
    dates = train.keys().tolist()
    dates.extend(test.keys().tolist())

    current_date = dates[-1]
    for i in range(steps):
        current_date += timedelta(days=1)
        dates.append(current_date)



    return {
        "train": train.tolist(),
        "test": test.tolist(),
        "forecast": pred,
        "metrics": metrics,
        "dates": [i.strftime("%Y-%m-%d") for i in dates],
        "counter": {"train":len(train.tolist()), "test":len(test.tolist()), "forecast":len(pred), "dates":len(dates)},
    }
