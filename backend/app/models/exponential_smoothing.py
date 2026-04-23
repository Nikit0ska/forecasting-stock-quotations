import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm

def forecast(train, val, test, forecast_days, alpha_range):
    y_train = train
    y_val = val
    y_test = test

    best_rmse = float('inf')
    best_predictions = None
    best_alpha = None

    # Перебор различных значений alpha
    for alpha in alpha_range:
        model = sm.tsa.SimpleExpSmoothing(y_train).fit(smoothing_level=alpha, optimized=False)
        predictions = model.forecast(forecast_days)  # Прогноз на forecast_days

        # Вычисление метрик для валидации
        rmse = np.sqrt(mean_squared_error(y_val[:forecast_days], predictions[:len(val)]))  # используем val для подбора
        if rmse < best_rmse:
            best_rmse = rmse
            best_predictions = predictions
            best_alpha = alpha

    # Финальные предсказания на forecast_days
    final_predictions = best_predictions
    rmse = np.sqrt(mean_squared_error(y_test[:forecast_days], final_predictions))
    mae = mean_absolute_error(y_test[:forecast_days], final_predictions)
    mape = np.mean(np.abs((y_test[:forecast_days] - final_predictions) / y_test[:forecast_days])) * 100

    return final_predictions, rmse, mae, mape, best_alpha