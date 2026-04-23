import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def forecast(train, val, test, forecast_days):
    y_train = train
    y_val = val
    y_test = test

    # Наивный прогноз: последняя котировка на все прогнозируемые дни
    naive_predictions = [y_train.iloc[-1]] * forecast_days  # прогноз на forecast_days дней

    # Вычисление метрик
    rmse = np.sqrt(mean_squared_error(y_test[:forecast_days], naive_predictions))
    mae = mean_absolute_error(y_test[:forecast_days], naive_predictions)
    mape = np.mean(np.abs((y_test[:forecast_days] - naive_predictions) / y_test[:forecast_days])) * 100

    return naive_predictions, rmse, mae, mape