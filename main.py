from data.data_handler import DataHandler
from portfolio import Portfolio
from trading_system import TradingSystem
from strategy import BasicStrategy

def main():
    data_handler = DataHandler(symbol="AAPL", interval="60min")
    df = data_handler.fetch_data()
    portfolio = Portfolio(risk_tolerance=1)
    strategy = BasicStrategy(portfolio)
    trading_system = TradingSystem(portfolio, strategy)

    trading_system.run_trading(df)

    # Check portfolio details
    print(portfolio.get_balance())
    print(portfolio.get_positions())
    print(portfolio.get_trade_history())

if __name__ == "__main__":
    main()
