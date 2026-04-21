def forecast(train, steps):
    last = train.iloc[-1]
    return [last] * steps
