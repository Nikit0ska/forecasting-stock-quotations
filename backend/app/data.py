import time

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

ASSETS = {
    "SBER": "Сбербанк",
    "GAZP": "Газпром",
    "LKOH": "Лукойл",
    "ROSN": "Роснефть",
    "NVTK": "Новатэк",
    "TATN": "Татнефть"
}


def load_series(asset: str, start_date="2025-10-05", max_rows=10000):
    if asset not in ASSETS:
        raise ValueError(f"Unknown asset: {asset}")

    url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{asset}.csv"

    params = {
        'from': start_date,
        'till': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
        'start': 0
    }

    all_dataframes = []
    start_position = 0
    batch_size = 100

    try:
        while True:
            current_url = f"{url}?from={params['from']}&till={params['till']}&start={start_position}&marketprice_board=1"

            df = pd.read_csv(
                current_url,
                sep=';',
                skiprows=2,
                encoding='windows-1251',
                skipfooter=4,
                engine='python'
            )

            if df.empty or len(df) == 0:
                break
            all_dataframes.append(df)

            if len(df) < batch_size:
                break

            start_position += len(df)

            if start_position >= max_rows:
                print(f"Достигнут максимальный лимит строк: {max_rows}")
                break

            time.sleep(0.1)

        if not all_dataframes:
            return pd.Series()

        full_df = pd.concat(all_dataframes, ignore_index=True)

        full_df['TRADEDATE'] = pd.to_datetime(full_df['TRADEDATE'])
        full_df = full_df.sort_values('TRADEDATE')

        if 'CLOSE' in full_df.columns:
            price_column = 'CLOSE'
        else:
            price_column = 'LEGALCLOSEPRICE'

        series = full_df.set_index('TRADEDATE')[price_column]

        series = series[~series.index.duplicated(keep='first')]
        series = series.sort_index()

        return series.dropna()

    except Exception as e:
        print(f"Error loading {asset}: {e}")
        return pd.Series()

def generate_test_data(asset: str, start: str):
    dates = pd.date_range(start=start, end=datetime.now(), freq='D')
    n = len(dates)

    base_prices = {
        "SBER": 300, "GAZP": 180, "LKOH": 7500,
        "ROSN": 600, "NVTK": 1700, "TATN": 350
    }

    base = base_prices.get(asset, 100)

    # Генерируем случайные данные с трендом
    np.random.seed(hash(asset) % 1000)
    noise = np.random.normal(0, 0.02, n).cumsum()
    trend = np.linspace(0, 0.5, n)

    prices = base * (1 + trend + noise)

    return pd.Series(prices, index=dates)