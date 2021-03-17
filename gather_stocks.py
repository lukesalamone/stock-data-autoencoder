import yfinance as yf
import json
import os
from pathlib import Path
import numpy as np

METADATA_PATH = "stock_data/metadata.json"
OUT_PATH = 'stock_data'

def download_files():
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    name = metadata['name']
    components_list = metadata['components']
    start_date = metadata['start_date']
    end_date = metadata['end_date']

    if not Path(OUT_PATH).exists():
        os.mkdir(OUT_PATH)

    print(f'\n\n-------- DOWNLOADING COMPONENTS FOR {name} --------\n\n')

    df = yf.download(tickers=components_list, start=start_date, end=end_date, threads=True)
    df['Date'] = df.index
    df = df[['Adj Close']].fillna(method='ffill')
    df.columns = [col[1] for col in df.columns]
    return df


def clean_data(stock_data):
    col_count = stock_data.count(axis=0).to_dict()

    unique, counts = np.unique([x for x in col_count.values()], return_counts=True)
    min_len = unique[np.argmax(counts)]

    symbols = {k:v for k,v in col_count.items() if v >= min_len}

    prices = [stock_data[c].to_numpy() for c in stock_data.columns if c in symbols]
    prices = [stock[-min_len:] for stock in prices]
    symbols = [k for k in symbols.keys()]

    return symbols, prices


def save_data(symbols, cleaned_data):
    for symbol, series in zip(symbols, cleaned_data):
        path = os.path.join(OUT_PATH, f'{symbol}.csv')
        np.savetxt(path, series, delimiter=',', fmt='%f')
        print(f'symbol {symbol} saved to file')


if __name__ == '__main__':
    # downloads stock data
    stock_data = download_files()

    # processes downloaded data to ensure consistent dates
    symbols, cleaned_data = clean_data(stock_data)

    # save to directory stock_data
    save_data(symbols, cleaned_data)
