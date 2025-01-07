import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import time, datetime
from calendar import monthrange

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from data.stock_data_service import StockDataService

class PreprocessData:
    def __init__(self, data_service: StockDataService):
        self.data_servive = data_service

    def preprocess_data(self, symbol, start_date, end_date, test_size = 0.15):
        raw_nested_data = self.data_servive.get_data(symbol, start_date, end_date)
        raw_data = [item for sublist in raw_nested_data for item in sublist]
        df = pd.DataFrame(raw_data)
        df.sort_values(['date', 'time'], ascending=[True, True])
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
        df['date'] = pd.to_datetime(df['date'])
        df = remove_spikes(df)
        df = self.add_technical_indicators(df)
        self.create_labels(df)
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.dayofweek + 1  # Add 1 so Monday=1 and Sunday=7
        df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour
        df.drop(columns=['symbol', 'date', 'time'], inplace=True)
        df.dropna(inplace=True)
        X_train, X_test, y_train, y_test = self.split_data(df, test_size)

        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #print(X_train_scaled)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def split_data(self, df, test_size):
        y = df['label']
        X = df.drop(columns=['label'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        return X_train, X_test, y_train, y_test

    def add_technical_indicators(self, df):
        #df['ma5'] = ta.sma(df['close'], length = 5)
        #df['ma10'] = ta.sma(df['close'], length = 10)
        df['ma20'] = ta.sma(df['close'], length = 20)
        df['ma50'] = ta.sma(df['close'], length = 50)
        #df['ma100'] = ta.sma(df['close'], length = 100)
        df['ma200'] = ta.sma(df['close'], length = 200)
        df['ema50'] = ta.ema(df['close'], length = 50)
        df['rsi'] = ta.rsi(df['close'], length = 14)
        df['obv'] = ta.obv(df['close'], df['volume'])
        df['atr'] = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        bb = ta.bbands(df['close'], length=20, std=2)
        return pd.concat([df, bb], axis=1)
    
        #new
        # for i in range(2, len(df) + 1):
        #     levels = self.calculate_fibonacci_levels(df.iloc[:i])
        #     for level_name, level_value in levels.items():
        #         df.at[df.index[i - 1], level_name] = level_value

        # df['roc'] = ta.roc(df['close'], length=10)

        # stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        # df = pd.concat([df, stoch], axis=1)

        # # Moving Average Convergence Divergence (MACD)
        
        # print(macd)
        

        #new

        

    def calculate_fibonacci_levels(self, df, period=17):
        # Filter the last 'period' days of data
        recent_data = df.tail(period)

        # Identify the swing high and swing low in the recent data
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()

        # Calculate the difference
        difference = swing_high - swing_low

        # Calculate Fibonacci levels
        fib_levels = {
            'Fibonacci Level 0%': swing_high,
            'Fibonacci Level 23.6%': swing_high - difference * 0.236,
            'Fibonacci Level 38.2%': swing_high - difference * 0.382,
            'Fibonacci Level 50%': swing_high - difference * 0.5,
            'Fibonacci Level 61.8%': swing_high - difference * 0.618,
            'Fibonacci Level 100%': swing_low,
        }

        return fib_levels

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
    
    def create_labels(self, df, threshhold=0.005, stop_loss_multiplier=1.5, take_profit_multiplier=3.0):
        labels = []
        for i in range(len(df) - 1):
            label = 0 # default to neutral
            stop_loss = df['close'].iloc[i] - (df['atr'].iloc[i] * stop_loss_multiplier)
            take_profit = df['close'].iloc[i] + (df['atr'].iloc[i] * take_profit_multiplier)
            for k in range(i + 1, len(df)):
                if not is_market_open(df['time'].iloc[k]):
                    break
                if df['close'].iloc[k] <= stop_loss:
                    label = 0
                    break
                if df['close'].iloc[k] >= take_profit:
                    label = 1
                    break
            labels.append(label)

        labels.append(0)
        df['label'] = labels

        buy_signals = df[df['label'] == 1]
        print("Buy signals:")
        print(buy_signals)
    
    def create_sequences(self, data, time_steps):
        sequences = []
        labels = []
        for i in range(len(data) - time_steps):
            # Extract the sequence of time_steps length
            sequence = data[i:i + time_steps]
            label = data[i + time_steps]  # The target label (next value after the sequence)
            sequences.append(sequence)
            labels.append(label)
        return np.array(sequences), np.array(labels)

def is_market_open(cur_time):
    market_open = time(9, 30)
    market_close = time(16, 0)
    return market_open <= cur_time <= market_close

def remove_spikes(df, threshold=1.05):
    # Iterate through DataFrame rows, excluding the first and last rows
    for i in range(1, len(df) - 1):
        # Check if current 'close' is more than `threshold` times higher or lower than the previous and next prices
        high = (df['close'].iloc[i] / df['close'].iloc[i - 1]) > threshold and (df['close'].iloc[i] / df['close'].iloc[i + 1]) > threshold
        low = (df['close'].iloc[i] / df['close'].iloc[i - 1]) < (1 / threshold) and (df['close'].iloc[i] / df['close'].iloc[i + 1]) < (1 / threshold)
        # If a spike is detected (either high or low)
        if high or low:
            # Replace the values with the average of the previous and next rows
            df.loc[i, 'open'] = (df['open'].iloc[i + 1] + df['open'].iloc[i - 1]) / 2
            df.loc[i, 'close'] = (df['close'].iloc[i + 1] + df['close'].iloc[i - 1]) / 2
            df.loc[i, 'high'] = (df['high'].iloc[i + 1] + df['high'].iloc[i - 1]) / 2
            df.loc[i, 'low'] = (df['low'].iloc[i + 1] + df['low'].iloc[i - 1]) / 2
    return df