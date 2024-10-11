import pandas as pd
import pandas_ta as ta
from datetime import time, datetime
from calendar import monthrange

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from data.database_interface import DatabaseInterface

class PreprocessData:
    def __init__(self, db: DatabaseInterface):
        self.db = db

    def preprocess_data(self, symbol, start_date, end_date, test_size = 0.15):
        start_date = f"{start_date[0]}-{str(start_date[1]).zfill(2)}-01"
        end_date = f"{end_date[0]}-{str(end_date[1]).zfill(2)}-{monthrange(end_date[0], end_date[1])[1]}"
        raw_data = self.db.get_data(symbol, start_date, end_date)
        df = pd.DataFrame(raw_data)
        df.sort_values(['date', 'time'], ascending=[True, True])
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
        df['date'] = pd.to_datetime(df['date'])
        df = self.add_technical_indicators(df)
        self.create_labels(df)
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.dayofweek + 1  # Add 1 so Monday=1 and Sunday=7
        df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour
        df.drop(columns=['symbol', 'date', 'time', 'open', 'high', 'low'], inplace=True)
        df.dropna(inplace=True)
        X_train, X_test, y_train, y_test = self.split_data(df, test_size)

        numerical_features = [
            'year', 'month', 'day', 'hour',
            'close', 'volume', 'ma20', 'ma50',
            'ma200', 'rsi', 'BBL_20_2.0', 'BBM_20_2.0',
            'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0'
        ]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[numerical_features])
        X_test_scaled = scaler.transform(X_test[numerical_features])

        print(X_train_scaled)

        return X_train_scaled, X_test_scaled, y_train, y_test


    # def preprocess_data(self, symbol, start_date, end_date):
    #     start_date = f"{start_date[0]}-{str(start_date[1]).zfill(2)}-01"
    #     end_date = f"{end_date[0]}-{str(end_date[1]).zfill(2)}-{monthrange(end_date[0], end_date[1])[1]}"
    #     raw_data = self.db.get_data(symbol, start_date, end_date)
    #     df = pd.DataFrame(raw_data)
    #     df.sort_values(['date', 'time'], ascending=[True, True])

    #     df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time  # convert time to proper time object
    #     df['date'] = pd.to_datetime(df['date'])  # convert date to datetime object

    #     print(df)
    #     df = self.add_technical_indicators(df)
    #     print(df)
    #     self.create_labels(df)
    #     self.normalize_data(df)

    #     df.drop(columns=['symbol', 'date', 'time'], inplace=True)
    #     df.dropna(inplace=True)

    #     print(df)
    #     return df

    def split_data(self, df, test_size):
        y = df['label']
        X = df.drop(columns=['label'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        return X_train, X_test, y_train, y_test

    def add_technical_indicators(self, df):
        df['ma20'] = ta.sma(df['close'], length = 20)
        df['ma50'] = ta.sma(df['close'], length = 50)
        df['ma200'] = ta.sma(df['close'], length = 200)

        df['rsi'] = ta.rsi(df['close'], length = 14)

        bb = ta.bbands(df['close'], length=20, std=2)
        return pd.concat([df, bb], axis=1)


    def normalize_data(self, df):
        scaler = StandardScaler()
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.dayofweek + 1 # Monday=0, Sunday=6
        df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour

        numerical_features = [
            'year', 'month', 'day', 'hour',
            'open', 'high', 'low',
            'close', 'volume', 'ma20', 'ma50',
            'ma200', 'rsi', 'BBL_20_2.0',  'BBM_20_2.0',
            'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
        ]
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    def create_labels(self, df, threshold=0.0121):
        labels = []
        for i in range(len(df) - 1):
            label = 0 # default to neutral
            for k in range(i + 1, len(df)):
                if not self.is_market_open(df['time'].iloc[k]):
                    break

                if (df['close'].iloc[k] / df['close'].iloc[i]) <= 1 - threshold/2:
                    label = 0
                    break
                if (df['close'].iloc[k] / df['close'].iloc[i]) >= 1 + threshold:
                    label = 1
                    break
            labels.append(label)

        labels.append(0)
        df['label'] = labels

        buy_signals = df[df['label'] == 1]
        print("Buy signals:")
        print(buy_signals)

    def is_market_open(self, cur_time):
        market_open = time(9, 30)
        market_close = time(16, 0)
        return market_open <= cur_time <= market_close