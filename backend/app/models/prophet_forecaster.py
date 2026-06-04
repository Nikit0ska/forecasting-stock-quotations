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
        full_series = full_series.sort_index()
        log_returns = np.log(full_series / full_series.shift(1)).dropna()

        # Подготавливаем данные для Prophet
        df = pd.DataFrame({
            'ds': log_returns.index,
            'y': log_returns.values
        })

        # ---- Подбор на train+val для walk-forward метрик ----
        train_val = df[df['ds'] <= val_end].copy()

        model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )

        model.fit(train_val)

        # ---- Прогноз на test ----
        # тестовые даты = только те, что реально есть в ряде
        test_dates = df[df['ds'] > val_end]['ds']
        future = pd.DataFrame({'ds': test_dates})
        forecast_test = model.predict(future)
        pred_logret = forecast_test['yhat'].values



        # ---- Переводим в цены ----
        last_price = full_series.loc[val_end]
        pred_prices = last_price * np.exp(np.cumsum(pred_logret))

        true_prices = full_series.loc[test_dates].values
        naive_errors = np.abs(true_prices - last_price)

        mae = np.mean(np.abs(true_prices - pred_prices))
        rmse = np.sqrt(np.mean((true_prices - pred_prices) ** 2))
        smape = np.mean(2 * np.abs(true_prices - pred_prices) / (np.abs(true_prices) + np.abs(pred_prices))) * 100
        mase = mae / np.mean(naive_errors) if np.mean(naive_errors) > 0 else np.nan

        # ---- Финальный прогноз на steps вперед ----
        future_steps = pd.date_range(start=full_series.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
        future_final = pd.DataFrame({'ds': future_steps})
        forecast_final = model.predict(future_final)
        final_pred_logret = forecast_final['yhat'].values
        final_pred_price = full_series.iloc[-1] * np.exp(np.cumsum(final_pred_logret))

        return {
            "pred": list(final_pred_price),
            "metrics": {"MAE": mae, "RMSE": rmse, "SMAPE": smape, "MASE": mase},
            "info": {"method": "Prophet"},
            "test_predictions": list(pred_prices),
            "test_actuals": list(true_prices),
            "test_naive_errors": list(naive_errors),
        }