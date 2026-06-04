from datetime import timedelta, datetime
from typing import Optional, Dict, Any

import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from app.data import load_series, ASSETS
from app.preprocessing import data_split
from app.metrics import regression_metrics
from app.schemas import ForecastResponse
from app.model_registry import get_forecaster
from app.models.ensemble_forecaster import EnsembleForecaster

# from app.models import naive, moving_average, arima, exponential_smoothing, auto_arima, wf_arima

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

@app.get("/forecast")
async def forecast(
    asset: str = Query(..., description="Тикер актива"),
    model1: str = Query("naive", description="Имя модели 1"),
    model2: Optional[str] = Query(None, description="Имя модели 2"),
    ensemble_weight: Optional[str] = Query(None, description="Вес через точку"),
    steps: int = Query(1, ge=1, description="Горизонт прогноза (дней)"),
    start_date: str = Query("2026-01-01", description="Дата начала данных"),
    train_ratio: Optional[str] = Query(None, description="Доля обучающих данных"),
    val_ratio: Optional[str] = Query(None, description="Доля валидаицонных данных"),
    test_ratio: Optional[str] = Query(None, description="Доля тестовых данных"),
    # дополнительные параметры модели можно передать через kwargs
    order_p: Optional[int] = Query(None),
    order_d: Optional[int] = Query(None),
    order_q: Optional[int] = Query(None),
    seq_length: Optional[int] = Query(None),
    epochs: Optional[int] = Query(None),
):
    # try:
    # Загрузка данных
    series = load_series(asset, start_date)
    if train_ratio is not None and val_ratio is not None and test_ratio is not None:
        train, val, test = data_split(series, validation_ratio=float(val_ratio) ,test_ratio=float(test_ratio))
    else:
        train, val, test = data_split(series)
    full_series = pd.concat([train, val, test])
    train_end = train.index[-1]
    val_end = val.index[-1]

    if model2 is not None:
        weights = None
        if ensemble_weight:
            weights = np.array([ensemble_weight, 1 - float(ensemble_weight)], dtype=np.float64)
        forecaster = EnsembleForecaster([model1, model2], weights)
    else:
        # Получаем модель
        forecaster = get_forecaster(model1)

    # Собираем дополнительные параметры, переданные пользователем
    extra_kwargs = {}
    # if model == "arima":
    #     if order_p is not None and order_d is not None and order_q is not None:
    #         extra_kwargs["order"] = (order_p, order_d, order_q)


    # Выполняем прогноз
    result = forecaster.forecast(full_series, train_end, val_end, steps, **extra_kwargs)

    # Готовим массив дат для ответа (как у вас было)
    dates = list(train.index) + list(val.index) + list(test.index)
    last_date = dates[-1]
    for i in range(1, steps + 1):
        dates.append(last_date + timedelta(days=i))

    return {
        "train": train.tolist(),
        "val": val.tolist(),
        "test": test.tolist(),
        "forecast": result["pred"],
        "metrics": result["metrics"],
        "info": result.get("info", {}),
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
    }

    # except ValueError as e:
    #     raise HTTPException(status_code=400, detail=str(e))
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
