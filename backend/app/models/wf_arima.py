# arima_forecaster.py
from statsmodels.tsa.arima.model import ARIMA
from app.models.base_forecaster import BaseForecaster
from app.models.wf_utils import walk_forward_evaluate

class ARIMAForecaster(BaseForecaster):
    def forecast(self, full_series, train_end, val_end, steps,
                 p_range=range(0,4), d_range=range(0,2), q_range=range(0,4), **kwargs):
        def arima_model_factory(history, params):
            order = (params['p'], params['d'], params['q'])
            try:
                model = ARIMA(history, order=order, trend='ct')
                fit = model.fit()
                return fit
            except:
                return None

        def arima_predict_func(model, steps):
            return model.forecast(steps=steps)

        param_list = [{'p':p,'d':d,'q':q} for p in p_range for d in d_range for q in q_range]
        best_params, _ = walk_forward_evaluate(
            full_series, train_end, val_end, steps,
            model_factory=arima_model_factory,
            predict_func=arima_predict_func,
            param_combinations=param_list,
            val_mode=True
        )

        def fixed_factory(history, params):
            return arima_model_factory(history, best_params)

        result = walk_forward_evaluate(
            full_series, train_end, val_end, steps,
            model_factory=fixed_factory,
            predict_func=arima_predict_func,
            param_combinations=None,
            val_mode=False
        )
        result['info'] = {'best_order': (best_params['p'], best_params['d'], best_params['q'])}
        return result