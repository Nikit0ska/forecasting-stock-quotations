# ma_forecaster.py
import numpy as np
import pandas as pd
from app.models.base_forecaster import BaseForecaster
from app.models.wf_utils import walk_forward_evaluate


class MAModel:
    def __init__(self, history, window):
        self.history = history
        self.window = window


class MAForecaster(BaseForecaster):
    def forecast(self, full_series, train_end, val_end, steps, window_range=range(3, 365), **kwargs):
        def ma_model_factory(history, params):
            if params is None:
                # при тестовом вызове params уже вшиты в замыкание
                # поэтому в тестовом режиме мы будем передавать фиксированную фабрику
                raise ValueError("params must be provided for training")
            window = params['window']
            return MAModel(history, window)

        def ma_predict_func(model, steps):
            # model типа MAModel
            if len(model.history) < model.window:
                # если недостаточно данных, возвращаем нули
                return np.zeros(steps)
            mean_ret = model.history.iloc[-model.window:].mean()
            return np.array([mean_ret] * steps)

        # Подбор окна на валидации
        param_list = [{'window': w} for w in window_range]
        best_params, _ = walk_forward_evaluate(
            full_series, train_end, val_end, steps,
            model_factory=ma_model_factory,
            predict_func=ma_predict_func,
            param_combinations=param_list,
            val_mode=True
        )

        # Тестирование с лучшим окном
        # Для тестового режима фиксируем параметры в фабрике
        def fixed_model_factory(history, params):
            # params игнорируется
            return MAModel(history, best_params['window'])

        result = walk_forward_evaluate(
            full_series, train_end, val_end, steps,
            model_factory=fixed_model_factory,
            predict_func=ma_predict_func,
            param_combinations=None,
            val_mode=False
        )
        result['info'] = {'best_window': best_params['window']}
        return result