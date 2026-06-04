import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from app.models.base_forecaster import BaseForecaster

class GRUForecaster(BaseForecaster):

    def __init__(self, window_size=20, hidden_units=50, epochs=20, batch_size=32):
        self.window_size = window_size
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size

    # -------------------------
    # Создание последовательностей для GRU
    # -------------------------
    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i + self.window_size])
            y.append(data[i + self.window_size])
        return np.array(X).reshape(-1, self.window_size, 1), np.array(y)

    # -------------------------
    # Построение модели
    # -------------------------
    def _build_model(self):
        model = Sequential([
            GRU(self.hidden_units, input_shape=(self.window_size, 1)),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    # -------------------------
    # Рекурсивный прогноз
    # -------------------------
    def _forecast_recursive(self, model, history, steps):
        seq = history[-self.window_size:].copy()
        preds = []
        for _ in range(steps):
            x = seq.reshape(1, self.window_size, 1)
            pred = model.predict(x, verbose=0)[0, 0]
            preds.append(pred)
            seq = np.append(seq[1:], pred)
        return np.array(preds)

    # -------------------------
    # Основной метод forecast
    # -------------------------
    def forecast(self, full_series, train_end, val_end, steps, **kwargs):

        # обновляем гиперпараметры, если переданы
        self.window_size = kwargs.get("window_size", self.window_size)
        self.hidden_units = kwargs.get("hidden_units", self.hidden_units)
        self.epochs = kwargs.get("epochs", self.epochs)
        self.batch_size = kwargs.get("batch_size", self.batch_size)

        full_series = full_series.sort_index()
        log_returns = np.log(full_series / full_series.shift(1)).dropna()

        train_res = log_returns[:val_end]
        test_res = log_returns[val_end + pd.Timedelta(days=1):]
        history = train_res.copy()
        last_price = full_series.loc[val_end]

        pred_prices = []
        true_prices = []
        naive_errors = []

        for i in range(len(test_res)):

            # создаём обучающие последовательности
            X_train, y_train = self._create_sequences(history.values)

            model = self._build_model()

            es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

            model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0,
                callbacks=[es]
            )

            # делаем прогноз на steps вперед
            pred_logret = self._forecast_recursive(model, history.values, steps)
            pred_price = last_price * np.exp(pred_logret[0])  # берём первый шаг

            true_price = full_series.loc[test_res.index[i]]

            pred_prices.append(pred_price)
            true_prices.append(true_price)
            naive_errors.append(abs(true_price - last_price))

            # обновляем историю
            history = pd.concat([history, pd.Series([test_res.iloc[i]], index=[test_res.index[i]])])
            last_price = full_series.loc[test_res.index[i]]

        # -------------------------
        # Метрики
        # -------------------------
        pred_prices = np.array(pred_prices)
        true_prices = np.array(true_prices)
        mae = np.mean(np.abs(true_prices - pred_prices))
        rmse = np.sqrt(np.mean((true_prices - pred_prices)**2))
        smape = np.mean(2 * np.abs(true_prices - pred_prices) / (np.abs(true_prices) + np.abs(pred_prices))) * 100
        mase = mae / np.mean(naive_errors) if np.mean(naive_errors) > 0 else np.nan

        # -------------------------
        # Финальный прогноз на steps дней вперед
        # -------------------------
        # X_train, y_train = self._create_sequences(history.values)
        # model = self._build_model()
        # es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        # model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=[es])
        #
        # final_pred_logret = self._forecast_recursive(model, history.values, steps)
        # final_pred_price = last_price * np.exp(np.cumsum(final_pred_logret))

        return {
            "pred": list(pred_prices)[:steps],
            "metrics": {"MAE": mae, "RMSE": rmse, "SMAPE": smape, "MASE": mase},
            "info": {"method": "GRU", "window_size": self.window_size, "hidden_units": self.hidden_units},
            "test_predictions": list(pred_prices),
            "test_actuals": list(true_prices),
            "test_naive_errors": list(naive_errors),
        }