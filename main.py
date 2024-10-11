from data.api_handler import ApiClient
from data.sqlite_database import SQLiteDatabase
from data.stock_data_service import StockDataService
from data.preprocessing_data import PreprocessData
from portfolio import Portfolio
from trading_system import TradingSystem
from learning_models.logistic_regression_model import LogisticRegressionModel
from learning_models.basic_model import BasicModel


def main():
    initial_balance = 100000
    portfolio = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
    basic_strategy = BasicModel(portfolio)
    api_client = ApiClient(interval="60min")
    data_base = SQLiteDatabase("persistence/stock_data.db")
    data_service = StockDataService(data_base, api_client)
    preprocessor = PreprocessData(data_base)

    trading_stocks = ["AAPL"]
    start_date = (2011, 1)
    end_date = (2024, 1)
    strategy = LogisticRegressionModel(portfolio, class_weight='balanced')

    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(trading_stocks[0], start_date, end_date)

    strategy.train_model(X_train, y_train)

    accuracy, predictions = strategy.evaluate_model(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    trading_system = TradingSystem(portfolio, strategy, data_service)
    trading_system.run_trading_simulation(trading_stocks, trading_start=start_date, trading_end=end_date, predictions=predictions)

    # Check portfolio details
    #print(portfolio.get_positions(), "\n")
    #print(*portfolio.get_trade_history(), sep='\n')
    print("Portfolio balance:", portfolio.get_balance(), ", Initial Balance:", initial_balance, ", Profit", portfolio.get_balance() / initial_balance, "\n")

if __name__ == "__main__":
    main()
