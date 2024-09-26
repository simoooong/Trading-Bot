import pandas as pd
from strategy import BasicStrategy
from portfolio import Portfolio

class TradingSystem:
    def __init__(self, portfolio: Portfolio, strategy: BasicStrategy):
        self.portfolio = portfolio
        self.strategy = strategy

    def run_trading(self, df: pd.DataFrame):
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(by='time')
        unique_dates = df['time'].dt.date.unique()
        
        for date in sorted(unique_dates):
            daily_data = df[df['time'].dt.date == date]
            closing_market_price = daily_data.iloc[-1]['close']

            self.process_day(daily_data, closing_market_price, date)

    def process_day(self, price_data_per_hour, closing_market_price, date,  symbol='AAPL'):
        for _, data in price_data_per_hour.iterrows():
            if self.strategy.should_enter_trade(symbol, data):
                self.strategy.enter_trade_long(symbol, data['close'], data['time'])

            if self.strategy.check_for_exit(symbol, data['close']):
                self.strategy.exit_trade_long(symbol, data['close'], data['time'])
            
        self.portfolio.close_all_positions(symbol, closing_market_price, data['time'])
