import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y

def recursive_forecast(model, history, window_size, steps):
    seq = history[-window_size:].copy()
    preds = []
    for _ in range(steps):
        x = seq.reshape(1, window_size, 1)
        pred = model.predict(x, verbose=0)[0, 0]
        preds.append(pred)
        seq = np.append(seq[1:], pred)
    return np.array(preds)

def walk_forward_lstm(
    full_series: pd.Series,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    steps: int,
    window_size: int = 20,
    hidden_units: int = 50,
    epochs: int = 20,
    batch_size: int = 32
):
    """
    Walk-forward для LSTM с рекурсивным прогнозом
    """
    full_series = full_series.sort_index()
    log_returns = np.log(full_series / full_series.shift(1)).dropna()

    test_series = log_returns[val_end + pd.Timedelta(days=1):]
    history = log_returns[:val_end].copy()
    last_known_price = full_series.loc[val_end]

    pred_prices = []
    true_prices = []
    naive_errors = []

    for i in range(len(test_series) - steps + 1):
        # Создание последовательностей для LSTM
        X_train, y_train = create_sequences(history.values, window_size)

        # Построение модели
        model = Sequential([
            LSTM(hidden_units, input_shape=(window_size, 1)),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

        # Рекурсивный прогноз
        pred_logret = recursive_forecast(model, history.values, window_size, steps)

        # Преобразуем лог-доходности в цену
        cum_logret = np.sum(pred_logret)
        pred_price = last_known_price * np.exp(cum_logret)
        true_price = full_series.loc[test_series.index[i + steps - 1]]

        pred_prices.append(pred_price)
        true_prices.append(true_price)
        naive_errors.append(abs(true_price - last_known_price))

        # Расширяем историю
        history = pd.concat([history, pd.Series([test_series.iloc[i]], index=[test_series.index[i]])])
        last_known_price = full_series.loc[test_series.index[i]]

    pred_prices = np.array(pred_prices)
    true_prices = np.array(true_prices)

    mae = np.mean(np.abs(true_prices - pred_prices))
    rmse = np.sqrt(np.mean((true_prices - pred_prices) ** 2))
    smape = np.mean(2 * np.abs(true_prices - pred_prices) / (np.abs(true_prices) + np.abs(pred_prices))) * 100
    naive_mae = np.mean(naive_errors)
    mase = mae / naive_mae if naive_mae > 0 else np.nan

    return {
        "pred": pred_prices,
        "true": true_prices,
        "metrics": {
            "MAE": mae,
            "RMSE": rmse,
            "SMAPE": smape,
            "MASE": mase
        }
    }