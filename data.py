import yfinance as yf
import numpy as np

class data:

    def __init__(self, ticker='AAPL', period='5y', interval='1d'):
        ticker_data = yf.Ticker(ticker)
        df = ticker_data.history(period, interval)
        self.data_matrix = df[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy()
        self.index = 0
        self.row_data = self.data_matrix[self.index]
        self.size = len(self.data_matrix)

    def get_tensor(self):
        return self.row_data
    
    def get_size(self):
        return self.size
    
    def next(self):
        self.index = self.index + 1
        self.row_data = self.data_matrix[self.index]

    def loop(self):
        self.index = 0
        self.row_data = self.data_matrix[self.index]