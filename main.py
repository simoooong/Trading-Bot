from data.api_handler import ApiClient
from data.sqlite_database import SQLiteDatabase
from data.stock_data_service import StockDataService
from data.preprocessing_data import PreprocessData
from portfolio import Portfolio
from trading_system import TradingSystem
from learning_models.strategy_interface import TradingStrategy
from learning_models.logistic_regression_model import LogisticRegressionModel
from learning_models.random_forest_model import RandomForestModel
from visualize_trading_results import visualize_performance, visualize_multiple_performance

import time


class TradingBot:
    def __init__(self, portfolio: Portfolio, strategy: TradingStrategy, initial_balance=100000):
        # Initialize the portfolio and service
        self.api_client = ApiClient(interval="60min")
        self.data_base = SQLiteDatabase("persistence/stock_data.db")
        self.data_service = StockDataService(self.data_base, self.api_client)
        self.preprocessor = PreprocessData(self.data_service)

        self.initial_balance = initial_balance
        self.portfolio = portfolio
        self.strategy = strategy
        self.stock_data = None

        # Define timeframe
        self.start_date = (2014, 1)
        self.end_date = (2024, 1)

        # Initialize the trading system
        self.trading_system = TradingSystem(self.portfolio, self.strategy, self.data_service)

    def setup_and_train_model(self, symbol, tune_hyperparameters = False):
        # Preprocess data and train the model
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess_data(symbol, self.start_date, self.end_date)
        print(f"_______________Test+train X = {len(X_train) + len(X_test)}")
        print(f"Test+train Y = {len(y_train) + len(y_test)}_______________")
        if tune_hyperparameters:
            self.strategy.tune_hyperparameters(X_train, y_train)
        self.strategy.train_model(X_train, y_train)
        return X_test, y_test

    def test_model(self, X_test, y_test):
        # Evaluate the model's accuracy
        accuracy, predictions =self.strategy.evaluate_model(X_test, y_test)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        return predictions

    def run_simulation(self, symbol, predictions):
        # Run trading simulation
        self.stock_data = self.trading_system.run_trading_simulation([symbol], self.start_date, self.end_date, predictions)

    def show_results(self):
        # Display results
        print(f"Portfolio balance: {self.portfolio.get_balance()}, Initial Balance: {self.initial_balance}, Profit: {self.portfolio.get_balance() / self.initial_balance:.4f}")

    def show_trading_logs(self):
        # Display trading logs
        print(*self.portfolio.get_trade_history(), sep='\n')

    def visualize_performance(self):
        # Display chart
        visualize_performance(self.stock_data, self.portfolio.get_trade_history(), self.initial_balance)

    def get_trading_logs(self):
        return self.portfolio.get_trade_history()
    
    def get_stock_data(self):
        return self.stock_data


def main():
    initial_balance = 100000
    trading_stocks = [
        # "SYK", "BSX", "ELV", "VRTX", "MDT",
        # "REGN", "BMY", "GILD", "CI", "ZTS",
        # "CVS", "HCA", "BDX", "MCK",
        "AMGN", "MRK", "DHR", "ISRG", "ABT",
        "TMO", "ABBV", "UNH", "LLY", "JNJ",
        "PFE"
    ]
    sector = "XLV"
    market = "SPY"

    """Trained model on sector"""
    portfolio_sector = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
    strategy = LogisticRegressionModel(portfolio_sector, class_weight='balanced')
    lr_sector = TradingBot(portfolio_sector, strategy, initial_balance)
    X_test, y_test = lr_sector.setup_and_train_model(sector, True)
    sector_predictions = lr_sector.test_model(X_test, y_test)

    """Train model on market"""
    portfolio_market = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
    strategy = LogisticRegressionModel(portfolio_market, class_weight='balanced')
    lr_market = TradingBot(portfolio_market, strategy, initial_balance)
    X_test, y_test = lr_market.setup_and_train_model(market, True)
    market_predictions = lr_market.test_model(X_test, y_test)


    for stock in trading_stocks:
        successful_execution = False
        while not successful_execution:
            try:
                trading_logs = {
                    "market trained model": None,
                    "sector trained model": None,
                    "stock trained model": None
                }
                """Stock trained model"""
                portfolio_stock = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
                strategy = LogisticRegressionModel(portfolio_stock, class_weight='balanced')
                lr_stock = TradingBot(portfolio_stock, strategy, initial_balance)
                X_test, y_test = lr_stock.setup_and_train_model(stock, True)
                stock_predictions = lr_stock.test_model(X_test, y_test)
                lr_stock.run_simulation(stock, stock_predictions)
                trading_logs["stock trained model"] = lr_stock.get_trading_logs()

                """Sector trained model"""
                portfolio_sector = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
                strategy = LogisticRegressionModel(portfolio_sector, class_weight='balanced')
                lr_sector = TradingBot(portfolio_sector, strategy, initial_balance)
                lr_sector.run_simulation(stock, sector_predictions)
                trading_logs["sector trained model"] = lr_sector.get_trading_logs()

                """Market trained model"""
                portfolio_market = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
                strategy = LogisticRegressionModel(portfolio_market, class_weight='balanced')
                lr_market = TradingBot(portfolio_market, strategy, initial_balance)
                lr_market.run_simulation(stock, market_predictions)
                trading_logs["market trained model"] = lr_market.get_trading_logs()

                visualize_multiple_performance(lr_stock.get_stock_data(), trading_logs, initial_balance)

                successful_execution = True
            except Exception as e:
                print(f"An error occurred with stock {stock}: {e}. Retrying...")
                time.sleep(60)

    '''
    Model: Logistic Regression
    Stock: JNJ
    '''
    # portfolio_lr_jnj = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
    # strategy = LogisticRegressionModel(portfolio_lr_jnj, class_weight='balanced')
    # lr_jnj = TradingBot(portfolio_lr_jnj, strategy, initial_balance)

    # X_test, y_test = lr_jnj.setup_and_train_model(trading_stocks[3], True)
    # predictions = lr_jnj.test_model(X_test, y_test)
    # lr_jnj.run_simulation(trading_stocks[3], predictions)
    # lr_jnj.show_results()
    # lr_jnj.visualize_performance()

    '''
    Model: Logistic Regression
    Stock: AAPL
    '''
    # portfolio_lr_aapl =  Portfolio(risk_tolerance=1, initial_balance=initial_balance)
    # strategy = LogisticRegressionModel(portfolio_lr_aapl, class_weight='balanced')
    # lr_aapl = TradingBot(portfolio_lr_aapl, strategy, initial_balance)

    # X_test, y_test = lr_aapl.setup_and_train_model(trading_stocks[0], False)
    # predictions = lr_aapl.test_model(X_test, y_test)
    # lr_aapl.run_simulation(trading_stocks[0], predictions)
    # lr_aapl.show_results()
    # lr_aapl.visualize_performance()

    '''
    Model: Random Forest
    Stock: JNJ
    '''
    # portfolio_rf_jnj =  Portfolio(risk_tolerance=1, initial_balance=initial_balance)
    # strategy = RandomForestModel(portfolio_rf_jnj, class_weight='balanced')
    # rf_aapl = TradingBot(portfolio_rf_jnj, strategy, initial_balance)

    # X_test, y_test = rf_aapl.setup_and_train_model(trading_stocks[0])
    # predictions = rf_aapl.test_model(X_test, y_test)
    # rf_aapl.run_simulation(trading_stocks[0], predictions)
    # rf_aapl.show_results()
    # rf_aapl.visualize_performance()

    '''
    Model: Random Forest
    Stock: AAPL
    '''
    # portfolio_rf_aapl =  Portfolio(risk_tolerance=1, initial_balance=initial_balance)
    # strategy = RandomForestModel(portfolio_rf_aapl, class_weight='balanced')
    # rf_aapl = TradingBot(portfolio_rf_aapl, strategy, initial_balance)

    # X_test, y_test = rf_aapl.setup_and_train_model(trading_stocks[1])
    # predictions = rf_aapl.test_model(X_test, y_test)
    # rf_aapl.run_simulation(trading_stocks[1], predictions)
    # rf_aapl.show_results()
    # rf_aapl.visualize_performance()

if __name__ == "__main__":
    main()
