# auto_arima_forecaster.py
# from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from app.models.base_forecaster import BaseForecaster
from app.models.wf_utils import walk_forward_evaluate

class AutoARIMAForecaster(BaseForecaster):
    def forecast(self, full_series, train_end, val_end, steps, **kwargs):
        # Параметры, управляющие поиском auto_arima (можно передать через kwargs)
        max_p = kwargs.get('max_p', 3)
        max_q = kwargs.get('max_q', 3)
        max_d = kwargs.get('max_d', 1)
        seasonal = kwargs.get('seasonal', False)
        stepwise = kwargs.get('stepwise', True)
        n_jobs = kwargs.get('n_jobs', 1)
        error_action = kwargs.get('error_action', 'ignore')
        suppress_warnings = kwargs.get('suppress_warnings', True)
        maxiter = kwargs.get('maxiter', 50)

        def arima_model_factory(history, params=None):
            # Параметры игнорируем – auto_arima сама всё подберёт
            try:
                model = auto_arima(
                    history,
                    start_p=0, start_q=0,
                    max_p=max_p, max_q=max_q, max_d=max_d,
                    seasonal=seasonal,
                    stepwise=stepwise,
                    n_jobs=n_jobs,
                    error_action=error_action,
                    suppress_warnings=suppress_warnings,
                    maxiter=maxiter
                )
                return model
            except Exception as e:
                print(f"AutoARIMA failed: {e}")
                return None

        def arima_predict_func(model, steps):
            return model.predict(n_periods=steps)  # pmdarima использует predict, не forecast

        # Для режима подбора на валидации передаём одну фиктивную комбинацию,
        # чтобы walk_forward_evaluate просто вызывал фабрику.
        best_params, _ = walk_forward_evaluate(
            full_series, train_end, val_end, steps,
            model_factory=arima_model_factory,
            predict_func=arima_predict_func,
            param_combinations=[{}],  # одна пустая комбинация
            val_mode=True
        )

        # Для тестового режима фиксированная фабрика (та же самая, параметров нет)
        def fixed_factory(history, params=None):
            return arima_model_factory(history)

        result = walk_forward_evaluate(
            full_series, train_end, val_end, steps,
            model_factory=fixed_factory,
            predict_func=arima_predict_func,
            param_combinations=None,
            val_mode=False
        )
        # Информация о лучшем порядке уже содержится в последней модели,
        # но её непросто извлечь. Можно вернуть заглушку.
        result['info'] = {'method': 'auto_arima'}
        return result