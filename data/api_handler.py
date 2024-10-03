import requests
import pandas as pd
import os
from dotenv import load_dotenv

class ApiClient:
    def __init__(self, interval):
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        self.interval = interval
        self.data = None

    def fetch_data_from_api(self, symbol, year, month):
        spec_month = f"{year}-{str(month).zfill(2)}"
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={self.interval}&month={spec_month}&outputsize=full&apikey={self.api_key}"
        
        req = requests.get(url)
        self.data = req.json()
        time_series = self.data['Time Series (60min)']

        # Transform the data into a list of dictionaries
        rows = []
        for time, price_data in time_series.items():
            date, hour = time.split()
            rows.append({
                'symbol': symbol,
                'date': date,
                'time': hour,
                'open': float(price_data['1. open']),
                'high': float(price_data['2. high']),
                'low': float(price_data['3. low']),
                'close': float(price_data['4. close']),
                'volume': int(price_data['5. volume'])
            })
        return rows
