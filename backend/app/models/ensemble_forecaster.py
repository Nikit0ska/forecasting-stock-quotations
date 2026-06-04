import numpy as np
import pandas as pd
from app.models.base_forecaster import BaseForecaster
from app.model_registry import get_forecaster


class EnsembleForecaster(BaseForecaster):
    """
    Ансамбль моделей с заданными весами.
    Принимает список названий моделей (model_names) и список весов (weights, по умолчанию равные).
    """

    def __init__(self, model_names, weights=None):
        self.model_names = model_names
        n = len(model_names)
        self.weights = np.array(weights) if weights is not None else np.ones(n) / n
        if len(self.weights) != n:
            raise ValueError("weights length must match model_names")

    def forecast(self, full_series, train_end, val_end, steps, **kwargs):
        # 1. Получаем прогнозы от всех базовых моделей
        base_results = []
        for name in self.model_names:
            forecaster = get_forecaster(name)
            res = forecaster.forecast(full_series, train_end, val_end, steps, **kwargs)
            base_results.append(res)

        # 2. Извлекаем test_predictions и test_actuals (они должны быть у всех моделей)
        # Убедимся, что все модели вернули эти списки и они одинаковой длины
        test_preds_list = []
        test_actuals = None
        for res in base_results:
            if "test_predictions" not in res or "test_actuals" not in res:
                raise ValueError(f"Model '{name}' did not return test_predictions. "
                                 f"All models must return test_predictions for ensemble to compute metrics.")
            test_preds_list.append(np.array(res["test_predictions"]))
            if test_actuals is None:
                test_actuals = np.array(res["test_actuals"])
            else:
                # Проверим, что фактические значения совпадают (ряды одни и те же)
                if not np.allclose(test_actuals, np.array(res["test_actuals"])):
                    raise ValueError("Actual test values mismatch between models")

        # 3. Взвешенное усреднение прогнозов
        ensemble_test_preds = np.average(test_preds_list, axis=0, weights=self.weights)

        # 4. Вычисляем метрики ансамбля на тесте
        true_prices = test_actuals
        pred_prices = ensemble_test_preds
        mae = np.mean(np.abs(true_prices - pred_prices))
        rmse = np.sqrt(np.mean((true_prices - pred_prices) ** 2))
        smape = np.mean(2 * np.abs(true_prices - pred_prices) /
                        (np.abs(true_prices) + np.abs(pred_prices))) * 100
        # Наивный бенчмарк: ошибка прогноза "цена не изменилась"
        naive_errors = np.abs(true_prices - np.roll(true_prices, 1))
        # Для первого элемента roll даст последний, но лучше взять предыдущую фактическую цену
        # Проще: наивный прогноз на шаге i = цена в день i-1 (last_known_price на начало шага)
        # Однако в нашем walk-forward мы хранили last_known_price, здесь у нас её нет.
        # Используем сдвиг фактических цен на 1 назад.
        naive_mae = np.mean(np.abs(true_prices[1:] - true_prices[:-1]))
        # но это не совсем тот наивный прогноз, который мы использовали раньше (там прогноз = текущая цена).
        # Для честности лучше, чтобы все модели возвращали и naive_errors, либо мы могли бы пересчитать,
        # используя last_known_price, которую надо сохранить. Для простоты пока примем naive_mae как среднее абсолютное изменение цены.
        # Однако в одиночных моделях мы использовали прогноз "цена не изменится", т.е. ошибка = |true_price - last_known_price|.
        # Здесь last_known_price для каждого шага теста равна цене в предыдущий день (val_end + i).
        # В массив true_prices входят цены через steps дней от этих точек. Без дополнительной информации трудно вычислить в точности.
        # Чтобы не усложнять, можно добавить в базовые модели возврат naive_errors или вычислять MASE иначе.
        # Предлагаю пока возвращать MASE = None, либо хранить также last_known_price для каждого шага.
        # Но для ответа пользователю я предложу более простой путь: ансамбль использует те же метрики, что и одиночные модели,
        # но для MASE мы можем взять среднюю ошибку наивного прогноза из любой базовой модели (они одинаковы).
        # Так как naive_errors во всех моделях одинаковы (фактические цены одни), мы можем извлечь naive_errors из первой попавшейся модели.
        # Для этого нужно, чтобы базовые модели тоже возвращали "test_naive_errors". Давай это добавим.
        # Вернёмся к этому после уточнения.

        # 5. Финальный прогноз (на будущее) – усредняем финальные прогнозы базовых моделей
        final_preds_list = [np.array(res["pred"]) for res in base_results]
        ensemble_final_pred = np.average(final_preds_list, axis=0, weights=self.weights)

        naive_errors = np.array(base_results[0]["test_naive_errors"])
        naive_mae = np.mean(naive_errors)
        mase = mae / naive_mae if naive_mae > 0 else np.nan

        return {
            "pred": ensemble_final_pred.tolist(),
            "metrics": {"MAE": mae, "RMSE": rmse, "SMAPE": smape, "MASE": mase},  # MASE пока None
            "info": {"ensemble": self.model_names, "weights": self.weights.tolist()},
            "test_predictions": ensemble_test_preds.tolist(),
            "test_actuals": test_actuals.tolist()
        }