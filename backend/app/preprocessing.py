def data_split(series,validation_ratio=0.1 ,test_ratio=0.2):
    n = len(series)
    split1 = int(n * (1 - validation_ratio - test_ratio))
    split2 = int(n * (1 - test_ratio))
    train = series[:split1]
    val = series[split1:split2]
    test = series[split2:]
    return train, val, test
