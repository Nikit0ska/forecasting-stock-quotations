import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast(prices, steps, trend='add', seasonal=None):
    series = pd.Series(prices)
    if len(series) < 10:
        return [float(series.iloc[-1])] * steps

    try:
        if seasonal and len(series) > 30:
            model = ExponentialSmoothing(
                series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=5  # 5 торговых дней в неделю
            )
        else:
            model = ExponentialSmoothing(series, trend=trend)

        fitted = model.fit()
        forecast = fitted.forecast(steps)

        return [float(x) for x in forecast]
    except:
        return [float(series.iloc[-1])] * steps
