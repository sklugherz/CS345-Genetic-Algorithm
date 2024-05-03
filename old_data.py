import yfinance as yf
import numpy as np

class data:

    def __init__(self, ticker='AAPL', period='5y', interval='1d'):
        ticker_data = yf.Ticker(ticker)
        df = ticker_data.history(period, interval)
        self.data_matrix = df[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy()
