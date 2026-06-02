from app.models.naive import NaiveForecaster
from app.models.moving_average import MAForecaster
from app.models.exponential_smoothing import ExpSmoothingForecaster
from app.models.auto_arima import AutoARIMAForecaster
from app.models.lstm_forecaster import LSTMForecaster
from app.models.prophet_forecaster import ProphetForecaster
from app.models.gru_forecaster import GRUForecaster


# Словарь доступных моделей
MODELS = {
    "naive": NaiveForecaster(),
    "ma": MAForecaster(),
    "exp": ExpSmoothingForecaster(),
    "arima": AutoARIMAForecaster(),
    "lstm": LSTMForecaster(),
    "prophet": ProphetForecaster(),
    "gru": GRUForecaster()
}

def get_forecaster(model_name: str):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    return MODELS[model_name]