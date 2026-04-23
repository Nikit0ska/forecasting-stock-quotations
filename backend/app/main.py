from datetime import timedelta, datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from app.data import load_series, ASSETS
from app.preprocessing import data_split
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
    train, val, test = data_split(series)

    if model == "naive":
        pred, rmse, mae, mape, = naive.forecast(train, val, test, forecast_days=steps)
    elif model == "ma":
        pred, rmse, mae, mape, best_window_size = moving_average.forecast(train, val, test, forecast_days=steps, window_range=range(3, 20))
    elif model == "arima":
        pred, rmse, mae, mape, best_order = arima.forecast(train, val, test, forecast_days=steps, p_range=range(1, 4), d_range=[1], q_range=range(0, 3))
    elif model == "exp":
        pred, rmse, mae, mape, best_alpha = exponential_smoothing.forecast(train, val, test, forecast_days=steps, alpha_range=np.arange(0.1, 1, 0.1))
    else:
        raise ValueError("Unknown model")

    dates = train.keys().tolist()
    dates.extend(val.keys().tolist())
    dates.extend(test.keys().tolist())

    current_date = dates[-1]
    for i in range(steps):
        current_date += timedelta(days=1)
        dates.append(current_date)



    return {
        "train": train.tolist(),
        "val": val.tolist(),
        "test": test.tolist(),
        "forecast": pred,
        "metrics": {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape
        },
        "dates": [i.strftime("%Y-%m-%d") for i in dates],

    }
