import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, Add, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping
from app.models.base_forecaster import BaseForecaster


# -------------------------------
# Transformer helper layers
# -------------------------------
def transformer_encoder(x, d_model, num_heads, ff_dim, dropout_rate=0.1):
    """Один блок энкодера Transformer: self-attention + feed-forward."""
    # Multi-head self-attention
    attn_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads
    )(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Feed-forward network
    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dense(d_model)(ffn)
    ffn = Dropout(dropout_rate)(ffn)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)
    return out2


def positional_encoding(length, depth):
    """Фиксированное синусно-косинусное позиционное кодирование."""
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (depths // 2)) / np.float32(depth))
    angle_rads = positions * angle_rates
    # sin для чётных индексов, cos для нечётных
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_enc = angle_rads[np.newaxis, ...]          # (1, length, depth)
    return tf.constant(pos_enc, dtype=tf.float32)


# -------------------------------
# Transformer Forecaster
# -------------------------------
class TransformerForecaster(BaseForecaster):

    def __init__(
        self,
        window_size=20,
        d_model=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        dropout_rate=0.1,
        epochs=20,
        batch_size=32
    ):
        self.window_size = window_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
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
    # MODEL (Transformer)
    # -------------------------
    def _build_model(self):
        input_seq = Input(shape=(self.window_size, 1))

        # Проекция 1-мерного входа в d_model
        x = Dense(self.d_model)(input_seq)                # (batch, window, d_model)

        # Добавляем позиционное кодирование
        pos_enc = positional_encoding(self.window_size, self.d_model)
        x = Lambda(lambda z: z + pos_enc)(x)

        # Несколько блоков энкодера
        for _ in range(self.num_layers):
            x = transformer_encoder(
                x, self.d_model, self.num_heads,
                self.ff_dim, self.dropout_rate
            )

        # Берём только последний временной шаг и предсказываем 1 значение
        x = Lambda(lambda z: z[:, -1, :])(x)              # (batch, d_model)
        output = Dense(1)(x)                              # (batch, 1)

        model = Model(inputs=input_seq, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        return model

    # -------------------------
    # RECURSIVE FORECAST
    # -------------------------
    def _forecast_recursive(self, model, history, steps):
        seq = np.array(history[-self.window_size:], dtype=float)
        preds = []
        for _ in range(steps):
            x = seq.reshape(1, self.window_size, 1)
            pred = model.predict(x, verbose=0)[0, 0]
            preds.append(pred)
            seq = np.concatenate([seq[1:], [pred]])
        return np.array(preds)   # ГАРАНТИЯ РАЗМЕРА steps

    # -------------------------
    # MAIN FORECAST
    # -------------------------
    def forecast(self, full_series, train_end, val_end, steps, **kwargs):
        # Обновляем параметры из kwargs (как в LSTM)
        self.window_size = kwargs.get('window_size', self.window_size)
        self.d_model = kwargs.get('d_model', self.d_model)
        self.num_heads = kwargs.get('num_heads', self.num_heads)
        self.num_layers = kwargs.get('num_layers', self.num_layers)
        self.ff_dim = kwargs.get('ff_dim', self.ff_dim)
        self.dropout_rate = kwargs.get('dropout_rate', self.dropout_rate)
        self.epochs = kwargs.get('epochs', self.epochs)
        self.batch_size = kwargs.get('batch_size', self.batch_size)

        full_series = full_series.sort_index()

        # Логарифмические доходности
        log_returns = np.log(full_series / full_series.shift(1)).dropna()

        # Данные для теста и стартовая история
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

            # Рекурсивный прогноз на steps шагов (лог-доходности)
            pred_log = self._forecast_recursive(
                model,
                history.values,
                steps
            )

            # Проверка размера (защита от багов)
            assert len(pred_log) == steps, \
                f'BUG: got {len(pred_log)} instead of {steps}'

            # Обратный переход к цене
            pred_price = last_price * np.exp(np.sum(pred_log))
            true_price = full_series.loc[test_series.index[i]]

            pred_prices.append(pred_price)
            true_prices.append(true_price)
            naive_errors.append(abs(true_price - last_price))

            # Добавляем реализовавшееся значение в историю
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
            'pred': list(pred_prices)[:steps],
            'metrics': {
                'MAE': mae,
                'RMSE': rmse,
                'SMAPE': smape,
                'MASE': mase
            },
            'info': {
                'method': 'Transformer',
                'window_size': self.window_size,
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'epochs': self.epochs
            },
            'test_predictions': list(pred_prices),
            'test_actuals': list(true_prices),
            'test_naive_errors': list(naive_errors),
        }