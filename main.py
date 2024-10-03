from data.api_handler import ApiClient
from data.sqlite_database import SQLiteDatabase
from data.stock_data_service import StockDataService
from portfolio import Portfolio
from trading_system import TradingSystem
from strategy import BasicStrategy

def main():
    initial_balance = 100000
    portfolio = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
    strategy = BasicStrategy(portfolio)
    api_client = ApiClient(interval="60min")
    data_base = SQLiteDatabase("persistence/stock_data.db")
    data_service = StockDataService(data_base, api_client)
    trading_system = TradingSystem(portfolio, strategy, data_service)

    trading_stocks = ["AAPL"]
    trading_system.run_trading_simulation(trading_stocks, trading_start=(2021, 1), trading_end=(2024,8))

    # Check portfolio details
    print("Portfolio balance:", portfolio.get_balance(), ", Initial Balance:", initial_balance, ", Profit", portfolio.get_balance() / initial_balance, "\n")
    print(portfolio.get_positions(), "\n")
    print(*portfolio.get_trade_history(), sep='\n')

if __name__ == "__main__":
    main()
