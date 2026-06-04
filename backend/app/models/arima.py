from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
def forecast(train, val, test, forecast_days, p_range, d_range, q_range):
    y_train = train
    y_val = val
    y_test = test

    best_rmse = float('inf')
    best_predictions = None
    best_order = None

    # Перебор различных параметров модели ARIMA
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(y_train, order=(p, d, q))
                    model_fit = model.fit()
                    predictions = model_fit.forecast(steps=forecast_days)  # Прогноз на forecast_days

                    # Вычисление метрик для валидации
                    rmse = np.sqrt(
                        mean_squared_error(y_val[:forecast_days], predictions[:len(val)]))  # используем val для подбора
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_predictions = predictions
                        best_order = (p, d, q)
                except:
                    continue  # Если модель не сходится, пропускаем

    # Финальные предсказания на forecast_days
    final_predictions = best_predictions
    rmse = np.sqrt(mean_squared_error(y_test[:forecast_days], final_predictions))
    mae = mean_absolute_error(y_test[:forecast_days], final_predictions)
    mape = np.mean(np.abs((y_test[:forecast_days] - final_predictions) / y_test[:forecast_days])) * 100

    return final_predictions, rmse, mae, mape, best_order
