import numpy as np
import pandas as pd
from app.models.base_forecaster import BaseForecaster

class NaiveForecaster(BaseForecaster):
    def forecast(self, full_series, train_end, val_end, steps, **kwargs):
        H = steps
        full_series = full_series.sort_index()

        # ----- Знаменатель MASE: MAE наивного метода на ОБУЧЕНИИ (шаг 1) -----
        train_series = full_series[:train_end]
        train_naive_errors = np.abs(train_series.diff().dropna()).values
        mase_denom = np.mean(train_naive_errors) if len(train_naive_errors) > 0 else 1e-6

        # ----- Тестовые даты -----
        test_dates = full_series.index[full_series.index > val_end]
        if len(test_dates) < H:
            raise ValueError(f"Тестовый период ({len(test_dates)}) меньше горизонта {H}")

        # ----- Собираем ошибки только на шаге H -----
        errors_h = []      # абсолютные ошибки
        actuals_h = []     # фактические цены (для SMAPE)
        preds_h = []       # прогнозные цены

        for i in range(len(test_dates) - H + 1):
            # Цена перед окном (база для наивного прогноза)
            if i == 0:
                price_before = full_series.loc[val_end]
            else:
                price_before = full_series.loc[test_dates[i-1]]

            # Фактическая цена на шаге H
            actual = full_series.loc[test_dates[i + H - 1]]
            pred = price_before
            error = abs(actual - pred)

            errors_h.append(error)
            actuals_h.append(actual)
            preds_h.append(pred)

        # ----- Итоговые метрики для горизонта H -----
        errors = np.array(errors_h)
        actual = np.array(actuals_h)
        pred = np.array(preds_h)

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors**2))
        smape = 100 * np.mean(2 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred) + 1e-8))
        mase = mae / mase_denom

        # ----- Финальный прогноз на H шагов (для поля "pred") -----
        final_pred = [full_series.iloc[-1]] * H

        return {
            "pred": final_pred,
            "metrics": {"MAE": float(mae), "RMSE": float(rmse), "SMAPE": float(smape), "MASE": float(mase)},
            "info": {"horizon": H, "num_windows": len(test_dates)-H+1},
            "test_predictions": None,
            "test_actuals": None,
            "test_naive_errors": list(train_naive_errors)
        }