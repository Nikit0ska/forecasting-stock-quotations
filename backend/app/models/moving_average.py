import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def forecast(train, val, test, forecast_days, window_range):
    y_train = train
    y_val = val
    y_test = test

    best_rmse = float('inf')
    best_predictions = None
    best_window_size = None

    # Перебор различных размеров окна
    for window_size in window_range:
        predictions = []
        for i in range(forecast_days):  # Прогноз на forecast_days
            window = train[-(window_size + i):-i] if i > 0 else train[-window_size:]
            predictions.append(np.mean(window))

        # Вычисление метрик для валидации
        rmse = np.sqrt(mean_squared_error(y_val[:forecast_days], predictions[:len(val)]))  # используем val для подбора
        if rmse < best_rmse:
            best_rmse = rmse
            best_predictions = predictions
            best_window_size = window_size

    # Финальные предсказания на forecast_days
    final_predictions = best_predictions
    rmse = np.sqrt(mean_squared_error(y_test[:forecast_days], final_predictions))
    mae = mean_absolute_error(y_test[:forecast_days], final_predictions)
    mape = np.mean(np.abs((y_test[:forecast_days] - final_predictions) / y_test[:forecast_days])) * 100

    return final_predictions, rmse, mae, mape, best_window_size