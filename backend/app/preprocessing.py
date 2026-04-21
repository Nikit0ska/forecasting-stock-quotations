def train_test_split(series, test_ratio=0.2):
    n = len(series)
    split = int(n * (1 - test_ratio))
    return series[:split], series[split:]
