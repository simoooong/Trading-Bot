import requests
import pandas as pd
import os
from dotenv import load_dotenv

class DataHandler:
    def __init__(self, symbol, interval):
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        self.symbol = symbol
        self.interval = interval
        self.data = None

    def fetch_data(self):
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={self.symbol}&interval={self.interval}&apikey={self.api_key}"
        req = requests.get(url)
        self.data = req.json()
        print (self.data)
        time_series = self.data['Time Series (60min)']

        # Transform the data into a list of dictionaries
        rows = []
        for time, price_data in time_series.items():
            rows.append({
                'time': time,
                'open': float(price_data['1. open']),
                'high': float(price_data['2. high']),
                'low': float(price_data['3. low']),
                'close': float(price_data['4. close']),
                'volume': int(price_data['5. volume'])
            })

        # Convert the list of dictionaries into a pandas DataFrame
        df = pd.DataFrame(rows)

        df['time'] = pd.to_datetime(df['time'])

        return df

    def get_data(self):
        return self.data
