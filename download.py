import yfinance as yf
import pandas as pd

def download_data(tickers, start_date, end_date):
    """
    Downloads historical stock data for the given tickers and date range.

    Parameters:
        tickers (list): List of stock tickers to download.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.    
    """

    # Download data using yfinance
    data = yf.download(tickers, start=start_date, end=end_date)
    adj_close = data['Adj Close']
    adj_close.to_csv('data/stock_data.csv')

    print("Data downloaded and saved to 'data/stock_data.csv'.")