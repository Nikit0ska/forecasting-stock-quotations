import pandas as pd
import numpy as np
from prophet import Prophet
from app.models.base_forecaster import BaseForecaster

class ProphetForecaster(BaseForecaster):

    def __init__(self, yearly_seasonality='auto', weekly_seasonality='auto', daily_seasonality=False):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality

    def forecast(self, full_series, train_end, val_end, steps, **kwargs):
        H = steps
        full_series = full_series.sort_index()

        # ----- Знаменатель MASE (одношаговый наивный на обучении) -----
        train_series = full_series[:train_end]
        train_naive_errors = np.abs(train_series.diff().dropna()).values
        mase_denom = np.mean(train_naive_errors) if len(train_naive_errors) > 0 else 1e-6

        # ----- Тестовые даты -----
        test_dates = full_series.index[full_series.index > val_end]
        if len(test_dates) < H:
            raise ValueError(f"Тестовый период ({len(test_dates)}) меньше горизонта {H}")

        # ----- Данные для Prophet (прогнозируем цены) -----
        df = pd.DataFrame({'ds': full_series.index, 'y': full_series.values})

        # ----- Хранилища ошибок только для шага H -----
        errors_h = []
        actuals_h = []
        preds_h = []

        # ----- Walk-forward, для каждого окна прогнозируем на H шагов, но берём только ошибку на шаге H -----
        for i in range(len(test_dates) - H + 1):
            window_start = test_dates[i]
            train_df = df[df['ds'] < window_start].copy()
            if len(train_df) < 2:
                continue

            # Обучаем Prophet
            model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality
            )
            if len(train_df) < 100:
                model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
            model.fit(train_df)

            # Прогноз на H шагов вперёд
            future_dates = [test_dates[i + h - 1] for h in range(1, H+1)]
            future = pd.DataFrame({'ds': future_dates})
            forecast = model.predict(future)
            pred_values = forecast['yhat'].values

            # Берём только шаг H
            pred = pred_values[H-1]
            actual = full_series.loc[future_dates[H-1]]
            error = abs(actual - pred)

            errors_h.append(error)
            actuals_h.append(actual)
            preds_h.append(pred)

        # ----- Итоговые метрики для горизонта H -----
        if len(errors_h) == 0:
            raise RuntimeError("Не удалось построить ни одного окна. Возможно, тестовый период слишком мал.")

        errors = np.array(errors_h)
        actual = np.array(actuals_h)
        pred = np.array(preds_h)

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors**2))
        smape = 100 * np.mean(2 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred) + 1e-8))
        mase = mae / mase_denom

        # ----- Финальный прогноз на H шагов (для поля "pred") -----
        final_model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )
        final_model.fit(df)
        future_steps = pd.date_range(start=full_series.index[-1] + pd.Timedelta(days=1), periods=H, freq='D')
        future_final = pd.DataFrame({'ds': future_steps})
        forecast_final = final_model.predict(future_final)
        final_pred = forecast_final['yhat'].values

        return {
            "pred": list(final_pred),
            "metrics": {"MAE": float(mae), "RMSE": float(rmse), "SMAPE": float(smape), "MASE": float(mase)},
            "info": {"method": "Prophet", "horizon": H, "num_windows": len(test_dates)-H+1},
            "test_predictions": None,
            "test_actuals": None,
            "test_naive_errors": list(train_naive_errors)
        }