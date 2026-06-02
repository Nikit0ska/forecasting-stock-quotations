import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from app.models.base_forecaster import BaseForecaster

class LSTMForecaster(BaseForecaster):

    def __init__(self, window_size=20, hidden_units=50, epochs=20, batch_size=32):
        self.window_size = window_size
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size

    # -------------------------
    # TRAIN SEQUENCES
    # -------------------------
    def _create_sequences(self, data):
        X, y = [], []

        for i in range(len(data) - self.window_size):
            X.append(data[i:i + self.window_size])
            y.append(data[i + self.window_size])

        X = np.array(X).reshape(-1, self.window_size, 1)
        y = np.array(y)

        return X, y

    # -------------------------
    # MODEL
    # -------------------------
    def _build_model(self):
        model = Sequential([
            LSTM(self.hidden_units, input_shape=(self.window_size, 1)),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    # -------------------------
    # CRITICAL FIX: FORECAST ALWAYS RETURNS SIZE = steps
    # -------------------------
    def _forecast_recursive(self, model, history, steps):
        seq = np.array(history[-self.window_size:], dtype=float)

        preds = []

        for _ in range(steps):
            x = seq.reshape(1, self.window_size, 1)

            pred = model.predict(x, verbose=0)[0, 0]

            preds.append(pred)

            seq = np.concatenate([seq[1:], [pred]])

        return np.array(preds)   # <<< ВОТ ТУТ ГАРАНТИЯ РАЗМЕРА

    # -------------------------
    # MAIN FORECAST
    # -------------------------
    def forecast(self, full_series, train_end, val_end, steps, **kwargs):

        self.window_size = kwargs.get("window_size", self.window_size)
        self.hidden_units = kwargs.get("hidden_units", self.hidden_units)
        self.epochs = kwargs.get("epochs", self.epochs)
        self.batch_size = kwargs.get("batch_size", self.batch_size)

        full_series = full_series.sort_index()

        log_returns = np.log(full_series / full_series.shift(1)).dropna()

        test_series = log_returns[val_end + pd.Timedelta(days=1):]
        history = log_returns[:val_end].copy()

        last_price = full_series.loc[val_end]

        pred_prices = []
        true_prices = []
        naive_errors = []

        # -------------------------
        # WALK-FORWARD LOOP
        # -------------------------
        for i in range(len(test_series)):

            X_train, y_train = self._create_sequences(history.values)

            model = self._build_model()

            model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0
            )

            # 🔥 FIX: pred ALWAYS = steps
            pred_log = self._forecast_recursive(
                model,
                history.values,
                steps
            )

            # safety check (чтобы больше никогда не ловить баг)
            assert len(pred_log) == steps, f"BUG: got {len(pred_log)} instead of {steps}"

            pred_price = last_price * np.exp(np.sum(pred_log))

            true_price = full_series.loc[test_series.index[i]]

            pred_prices.append(pred_price)
            true_prices.append(true_price)

            naive_errors.append(abs(true_price - last_price))

            history = pd.concat([
                history,
                pd.Series([test_series.iloc[i]], index=[test_series.index[i]])
            ])

            last_price = full_series.loc[test_series.index[i]]

        # -------------------------
        # METRICS
        # -------------------------
        pred_prices = np.array(pred_prices)
        true_prices = np.array(true_prices)

        mae = np.mean(np.abs(true_prices - pred_prices))
        rmse = np.sqrt(np.mean((true_prices - pred_prices) ** 2))
        smape = np.mean(
            2 * np.abs(true_prices - pred_prices) /
            (np.abs(true_prices) + np.abs(pred_prices))
        ) * 100

        mase = mae / np.mean(naive_errors) if np.mean(naive_errors) > 0 else np.nan


        return {
            "pred": list(pred_prices)[:steps],
            "metrics": {
                "MAE": mae,
                "RMSE": rmse,
                "SMAPE": smape,
                "MASE": mase
            },
            "info": {
                "method": "LSTM",
                "window_size": self.window_size,
                "hidden_units": self.hidden_units,
                "epochs": self.epochs
            }
        }