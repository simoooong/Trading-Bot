from data.api_handler import ApiClient
from data.sqlite_database import SQLiteDatabase
from data.stock_data_service import StockDataService
from data.preprocessing_data import PreprocessData
from portfolio import Portfolio
from trading_system import TradingSystem
from learning_models.strategy_interface import TradingStrategy
from learning_models.logistic_regression_model import LogisticRegressionModel


class TradingBot:
    def __init__(self, portfolio: Portfolio, strategy: TradingStrategy, initial_balance=100000):
        # Initialize the portfolio and services
        self.api_client = ApiClient(interval="60min")
        self.data_base = SQLiteDatabase("persistence/stock_data.db")
        self.data_service = StockDataService(self.data_base, self.api_client)
        self.preprocessor = PreprocessData(self.data_base, self.data_service)
        
        self.initial_balance = initial_balance
        self.portfolio = portfolio
        self.strategy = strategy

        # Define timeframe
        self.start_date = (2014, 1)
        self.end_date = (2024, 1)

        # Initialize the trading system
        self.trading_system = TradingSystem(self.portfolio, self.strategy, self.data_service)

    def setup_and_train_model(self, symbol):
        # Preprocess data and train the model
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess_data(symbol, self.start_date, self.end_date)
        self.strategy.train_model(X_train, y_train)
        return X_test, y_test
    
    def test_model(self, X_test, y_test):
        # Evaluate the model's accuracy
        accuracy, predictions =self.strategy.evaluate_model(X_test, y_test)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        return predictions
    
    def run_simulation(self, symbol, predictions):
        # Run trading simulation
        self.trading_system.run_trading_simulation([symbol], self.start_date, self.end_date, predictions)

    def show_results(self):
        # Display results
        print(f"Portfolio balance: {self.portfolio.get_balance()}, Initial Balance: {self.initial_balance}, Profit: {self.portfolio.get_balance() / self.initial_balance:.4f}")

    def show_trading_logs(self):
        # Display trading logs
        print(*self.portfolio.get_trade_history(), sep='\n')


def main():
    initial_balance = 100000
    trading_stocks = ["JNJ"]

    portfolio = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
    strategy = LogisticRegressionModel(portfolio, class_weight='balanced')

    bot = TradingBot(portfolio, strategy, initial_balance)
    X_test, y_test = bot.setup_and_train_model(trading_stocks[0])
    predictions = bot.test_model(X_test, y_test)
    bot.run_simulation(trading_stocks[0], predictions)
    bot.show_results()

if __name__ == "__main__":
    main()
