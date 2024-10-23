import pandas as pd
from learning_models.strategy_interface import TradingStrategy
from portfolio import Portfolio
from data.stock_data_service import StockDataService
from data.preprocessing_data import is_market_open

class TradingSystem:
    def __init__(self, portfolio: Portfolio, strategy: TradingStrategy, data_service: StockDataService):
        self.portfolio = portfolio
        self.strategy = strategy
        self.data_service = data_service

    def run_trading_simulation(self, trading_stocks, trading_start, trading_end, predictions = None):
        data = {}

        for symbol in trading_stocks:
            data[symbol] = self.data_service.get_data(symbol, trading_start, trading_end)

        return self.simulate_trading(data, predictions)

    def simulate_trading(self, data, predictions):
        combined_data = []
        for symbol, entries in data.items():
            for month_entry in entries:
                for day_entry in month_entry:
                    combined_entry = day_entry.copy()
                    combined_data.append(combined_entry)
        
        df = pd.DataFrame(combined_data)

        if predictions is not None:
            n = len(predictions)
            df = df[-n:]

        df['predictions'] = predictions
        print(df)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.sort_values(by='date', inplace=True)
        for day, daily_data in df.groupby('date'):
            #print(f"Processing data for {day}")
            self.process_day(day, daily_data)

        return df
        

    def process_day(self, day, daily_data):
        daily_data_sorted = daily_data.sort_values(by='time')
        daily_data_sorted['time'] = pd.to_datetime(daily_data_sorted['time'], format='%H:%M:%S').dt.time
        filtered_data = daily_data_sorted[daily_data_sorted['time'].apply(is_market_open)]
        for _, row in filtered_data.iterrows():
            symbol = row['symbol']
            #print(f"Processing {symbol} at {row['time']} with open={open_price}, close={close_price}")

            if self.strategy.should_enter_trade(symbol, row):
                self.strategy.enter_trade_long(symbol, row['close'], f"{day}-{row['time']}")

            if self.strategy.check_for_exit(symbol, row['close']):
                self.strategy.exit_trade_long(symbol, row['close'], f"{day}-{row['time']}")

        self.portfolio.close_all_positions(symbol, filtered_data.iloc[-1]['close'], f"{day}-{row['time']}")
