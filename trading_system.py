import pandas as pd
from learning_models.strategy_interface import TradingStrategy
from portfolio import Portfolio
from data.stock_data_service import StockDataService

class TradingSystem:
    def __init__(self, portfolio: Portfolio, strategy: TradingStrategy, data_service: StockDataService):
        self.portfolio = portfolio
        self.strategy = strategy
        self.data_service = data_service

    def run_trading_simulation(self, trading_stocks, trading_start, trading_end, predictions = None):
        data = {}

        start_year, start_month = trading_start
        end_year, end_month = trading_end

        for symbol in trading_stocks:
            data[symbol] = []

            current_year = start_year
            current_month = start_month

            while (current_year, current_month) <= (end_year, end_month):
                monthly_data = self.data_service.get_monthly_data(symbol, current_year, current_month)

                if monthly_data is not None:
                    data[symbol].append(monthly_data)
                
                if current_month == 12:
                    current_month = 1
                    current_year += 1
                else:
                    current_month += 1
            
        self.simulate_trading(data, predictions)

    def simulate_trading(self, data, predictions):
        combined_data = []
        for symbol, entries in data.items():
            for month_entry in entries:
                for day_entry in month_entry:
                    combined_entry = day_entry.copy()
                    combined_data.append(combined_entry)
        
        df = pd.DataFrame(combined_data)

        n = len(predictions)
        df = df[-n:]

        df['predictions'] = predictions
        print(df)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.sort_values(by='date', inplace=True)
        for day, daily_data in df.groupby('date'):
            #print(f"Processing data for {day}")
            self.process_day(day, daily_data)

    def process_day(self, day, daily_data):
        daily_data_sorted = daily_data.sort_values(by='time')

        for _, row in daily_data_sorted.iterrows():
            symbol = row['symbol']
            #print(f"Processing {symbol} at {row['time']} with open={open_price}, close={close_price}")

            if self.strategy.should_enter_trade(symbol, row):
                self.strategy.enter_trade_long(symbol, row['close'], f"{day}-{row['time']}")

            if self.strategy.check_for_exit(symbol, row['close']):
                self.strategy.exit_trade_long(symbol, row['close'], f"{day}-{row['time']}")

        self.portfolio.close_all_positions(symbol, daily_data_sorted.iloc[-1]['close'], f"{day}-{row['time']}")
