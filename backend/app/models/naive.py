# naive_forecaster.py
import numpy as np
import pandas as pd
from app.models.base_forecaster import BaseForecaster

class NaiveForecaster(BaseForecaster):
    def forecast(self, full_series, train_end, val_end, steps, **kwargs):
        # Преобразуем в лог-доходности (для единообразия, но наивный прогноз: доходность=0)
        log_returns = np.log(full_series / full_series.shift(1)).dropna()
        test_start = val_end + pd.Timedelta(days=1)
        test_series = log_returns[test_start:]

        # Имитируем walk-forward по тесту: прогноз на каждом шаге – нулевая доходность
        # Восстанавливаем цены, считаем ошибки
        errors_mae = []
        errors_rmse = []
        last_price = full_series.loc[val_end]  # цена на конец валидации

        # Для MASE считаем ошибки наивного прогноза "цена не изменится" (то же самое)
        naive_errors = []
        pred_prices = []
        true_prices = []
        for i, (date, true_logret) in enumerate(test_series.items()):
            # Прогноз доходности = 0
            pred_price = last_price * np.exp(0)
            pred_prices.append(pred_price)
            true_price = full_series.loc[date]  # фактическая цена сегодня
            true_prices.append(true_price)
            # Ошибка
            mae = abs(true_price - pred_price)
            errors_mae.append(mae)
            errors_rmse.append(mae ** 2)
            naive_errors.append(abs(true_price - last_price))
            last_price = true_price

        # Метрики
        mae_val = np.mean(errors_mae)
        rmse_val = np.sqrt(np.mean(errors_rmse))
        smape_val = np.mean(2 * np.array(errors_mae) / (np.array([full_series.loc[date] for date in test_series.index]) + last_price)) * 100  # упрощённо; в реальности лучше пошагово
        mase_val = mae_val / np.mean(naive_errors) if np.mean(naive_errors) > 0 else 0

        # Финальный прогноз от последней известной цены
        final_last_price = full_series.iloc[-1]
        pred = [final_last_price] * steps  # цена не меняется

        return {
            "pred": pred,
            "metrics": {"MAE": mae_val, "RMSE": rmse_val, "SMAPE": smape_val, "MASE": mase_val},
            "info": {},
            "test_predictions": pred_prices,
            "test_actuals": true_prices,
            "test_naive_errors": list(naive_errors),
        }