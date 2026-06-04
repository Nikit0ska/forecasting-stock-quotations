import numpy as np
import pandas as pd

def walk_forward_evaluate(
    full_series: pd.Series,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    steps: int,
    model_factory,          # (history, params) -> обученная модель (любой объект)
    predict_func,           # (model, steps) -> np.array прогнозов лог-доходностей
    param_combinations=None,# список словарей параметров для подбора
    val_mode=True           # True – подбор на валидации, False – тест с фикс. параметрами
):
    full_series = full_series.sort_index()
    log_returns = np.log(full_series / full_series.shift(1)).dropna()
    if val_mode:
        train_series = log_returns[:train_end]
        val_series = log_returns[train_end + pd.Timedelta(days=1):val_end]
        best_params = None
        best_score = np.inf
        for params in param_combinations:
            errors = []
            history = train_series.copy()
            # идём по дням валидации, для каждого делаем прогноз на steps дней
            for i in range(len(val_series) - steps + 1):
                t_date = val_series.index[i]
                # обучаем модель
                model = model_factory(history, params)
                if model is None:
                    continue
                # прогноз на steps
                pred_logret = predict_func(model, steps)
                actual_logret = val_series.iloc[i:i+steps].values
                if len(pred_logret) == len(actual_logret):
                    rmse = np.sqrt(np.mean((actual_logret - pred_logret)**2))
                    errors.append(rmse)
                # расширяем историю
                history = pd.concat([history, pd.Series([val_series.iloc[i]], index=[t_date])])
            if errors:
                avg_rmse = np.mean(errors)
                if avg_rmse < best_score:
                    best_score = avg_rmse
                    best_params = params


        return best_params, best_score
    else:
        # Тестовый период
        test_series = log_returns[val_end + pd.Timedelta(days=1):]
        history = log_returns[:val_end].copy()
        last_known_price = full_series.loc[val_end]
        pred_prices = []
        true_prices = []
        naive_errors = []

        # for i in range(len(test_series) - steps + 1):
        for i in range(len(test_series)):
            t_date = test_series.index[i]
            model = model_factory(history, None)  # params уже встроены
            if model is None:
                continue
            pred_logret = predict_func(model, steps)
            cum_logret = np.sum(pred_logret)
            pred_price = last_known_price * np.exp(cum_logret)
            # true_price = full_series.loc[test_series.index[i + steps - 1]]
            true_price = full_series.loc[test_series.index[i]]
            pred_prices.append(pred_price)
            true_prices.append(true_price)
            naive_errors.append(abs(true_price - last_known_price))
            history = pd.concat([history, pd.Series([test_series.iloc[i]], index=[t_date])])
            last_known_price = full_series.loc[t_date]

        pred_prices = np.array(pred_prices)
        true_prices = np.array(true_prices)
        mae = np.mean(np.abs(true_prices - pred_prices))
        rmse = np.sqrt(np.mean((true_prices - pred_prices)**2))
        smape = np.mean(2 * np.abs(true_prices - pred_prices) / (np.abs(true_prices) + np.abs(pred_prices))) * 100
        naive_mae = np.mean(naive_errors)
        mase = mae / naive_mae if naive_mae > 0 else np.nan

        # Финальный прогноз
        final_model = model_factory(log_returns, None)
        final_pred_logret = predict_func(final_model, steps)
        final_last_price = full_series.iloc[-1]
        final_pred_price = final_last_price * np.exp(np.cumsum(final_pred_logret))

        test_returns = log_returns[val_end + pd.Timedelta(days=1):]
        print("freq =", log_returns.index.freq)
        print("inferred_freq =", log_returns.index.inferred_freq)
        return {
            "pred": list(final_pred_price),
            "metrics": {"MAE": mae, "RMSE": rmse, "SMAPE": smape, "MASE": mase},
            "info": {},
            "test_predictions": list(pred_prices),
            "test_actuals": list(true_prices),
            "test_naive_errors": list(naive_errors),
        }