import numpy as np

def forecast(train, steps, window=5):
    value = np.mean(train[-window:])
    return [value] * steps
