from statsmodels.tsa.arima.model import ARIMA

def forecast(train, steps, order=(1, 1, 1)):
    model = ARIMA(train, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=steps)
    return forecast.astype(float).tolist()
