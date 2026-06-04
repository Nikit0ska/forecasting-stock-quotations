# exp_forecaster.py
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from app.models.base_forecaster import BaseForecaster
from app.models.wf_utils import walk_forward_evaluate


class ExpSmoothingForecaster(BaseForecaster):
    def forecast(self, full_series, train_end, val_end, steps,
                 trend='add',  # 'add', 'mul', None
                 damped_trend=True,
                 alpha_range=None,  # если None – подбираем автоматически
                 **kwargs):
        if alpha_range is None:
            alpha_range = np.arange(0.1, 1.0, 0.1)

        def ets_model_factory(history, params):
            alpha = params['alpha']
            try:
                # Если trend=None, то damped_trend игнорируется
                model = ExponentialSmoothing(history, trend=trend, damped_trend=damped_trend if trend else False)
                fit = model.fit(smoothing_level=alpha)
                return fit
            except:
                return None

        def predict_func(model, steps):
            return model.forecast(steps)

        param_list = [{'alpha': a} for a in alpha_range]

        best_params, _ = walk_forward_evaluate(
            full_series, train_end, val_end, steps,
            model_factory=ets_model_factory,
            predict_func=predict_func,
            param_combinations=param_list,
            val_mode=True
        )

        # Тестирование
        def fixed_factory(history, params):
            return ets_model_factory(history, best_params)

        result = walk_forward_evaluate(
            full_series, train_end, val_end, steps,
            model_factory=fixed_factory,
            predict_func=predict_func,
            param_combinations=None,
            val_mode=False
        )
        result['info'] = best_params
        # print(best_params)
        return result