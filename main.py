from data.api_handler import ApiClient
from data.sqlite_database import SQLiteDatabase
from data.stock_data_service import StockDataService
from data.preprocessing_data import PreprocessData
from portfolio import Portfolio
from trading_system import TradingSystem
from learning_models.strategy_interface import TradingStrategy
from learning_models.logistic_regression_model import LogisticRegressionModel
from learning_models.random_forest_model import RandomForestModel
from learning_models.svm import SupportVectorMachineModel
from learning_models.lstm import LSTMModel
from visualize_trading_results import visualize_performance, visualize_multiple_performance

import time
import numpy as np


class TradingBot:
    def __init__(self, portfolio: Portfolio, strategy: TradingStrategy, kelly_criteria = True, initial_balance=100000):
        # Initialize the portfolio and service
        self.api_client = ApiClient(interval="60min")
        self.data_base = SQLiteDatabase("persistence/stock_data.db")
        self.data_service = StockDataService(self.data_base, self.api_client)
        self.preprocessor = PreprocessData(self.data_service)

        self.kelly_criteria = kelly_criteria
        self.initial_balance = initial_balance
        self.portfolio = portfolio
        self.strategy = strategy
        self.stock_data = None
        self.x_train = None
        self.y_train = None

        # Define timeframe
        self.start_date = (2014, 1)
        self.end_date = (2024, 1)

        # Initialize the trading system
        self.trading_system = TradingSystem(self.portfolio, self.strategy, self.data_service)

    def setup_and_train_model(self, symbol, X_train = None, y_train = None, tune_hyperparameters = False):
        # Preprocess data and train the model
        X_train_loc, X_test, y_train_loc, y_test = self.preprocessor.preprocess_data(symbol, self.start_date, self.end_date)
        if X_train is None or y_train is None:
            self.x_train = X_train_loc
            self.y_train = y_train_loc
        else:
            self.x_train = X_train
            self.y_train = y_train

        if tune_hyperparameters:
            self.strategy.tune_hyperparameters(self.x_train, self.y_train)
        self.strategy.train_model(self.x_train, self.y_train)
        return X_test, y_test

    def test_model(self, X_test, y_test):
        # Evaluate the model's accuracy
        accuracy, predictions, probablities = self.strategy.evaluate_model(X_test, y_test)
        #print(f"Model accuracy: {accuracy * 100:.2f}%")
        if self.kelly_criteria:
            return predictions, probablities
        else:
            return predictions, None

    def run_simulation(self, symbol, predictions, probabilties):
        # Run trading simulation
        self.stock_data = self.trading_system.run_trading_simulation([symbol], self.start_date, self.end_date, predictions, probabilties)

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
    
    def get_training_data(self):
        return self.x_train, self.y_train

def generate_strategy(model_name, portfolio, class_weight):
    if model_name == "LR":
        return LogisticRegressionModel(portfolio, class_weight)
    elif model_name == "RF":
        return RandomForestModel(portfolio, class_weight)
    elif model_name == "LSTM":
        return LSTMModel(portfolio)
    elif model_name == "SVM":
        return SupportVectorMachineModel(portfolio, class_weight)

def main():
    '''Model context'''
    model_name = "LR"
    initial_balance = 100000
    tune_hyperparameters = True
    trained_on_all = False
    save_figures = False
    save_dir = "figures"

    trading_stocks = [
        "SYK", "ISRG", "LLY",
        "JNJ", "DHR", "TMO",
        "PFE", "BMY", "CVS"
        
    ]
    sector = "XLV"
    market = "SPY"

    """Trained model on sector"""
    portfolio_sector = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
    strategy = generate_strategy(model_name, portfolio_sector, class_weight='balanced')
    lr_sector = TradingBot(portfolio_sector, strategy, initial_balance)
    X_test, y_test = lr_sector.setup_and_train_model(sector, None, None, tune_hyperparameters)
    sector_predictions, sector_probabilties = lr_sector.test_model(X_test, y_test)

    """Train model on market"""
    portfolio_market = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
    strategy = generate_strategy(model_name, portfolio_market, class_weight='balanced')
    lr_market = TradingBot(portfolio_market, strategy, initial_balance)
    X_test, y_test = lr_market.setup_and_train_model(market, None, None, tune_hyperparameters)
    market_predictions, market_probabilties = lr_market.test_model(X_test, y_test)

    if trained_on_all:
        X_trading_data = []
        Y_trading_data = []

        X_train, y_train = lr_sector.get_training_data()
        X_trading_data.append(X_train)
        Y_trading_data.append(y_train)

        X_train, y_train = lr_market.get_training_data()
        X_trading_data.append(X_train)
        Y_trading_data.append(y_train)

        for stock in trading_stocks:
            successful_execution = False
            while not successful_execution:
                try:
                    """Stock trained model"""
                    portfolio_stock = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
                    strategy = LogisticRegressionModel(portfolio_stock, class_weight='balanced')
                    lr_stock = TradingBot(portfolio_stock, strategy, initial_balance)
                    lr_stock.setup_and_train_model(stock, None, None, False)
                    X_train, y_train = lr_stock.get_training_data()
                    X_trading_data.append(X_train)
                    Y_trading_data.append(y_train)
                    successful_execution = True
                except Exception as e:
                    print(f"An error occurred with stock {stock}: {e}. Retrying...")
                    time.sleep(60)

        X_combined = np.concatenate(X_trading_data, axis=0)
        Y_combined = np.concatenate(Y_trading_data, axis=0)

        """Train model on all data"""
        print("______________Start training the big shit____________")
        portfolio_all = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
        strategy = generate_strategy(model_name, portfolio_all, class_weight='balanced')
        lr_all = TradingBot(portfolio_all, strategy, initial_balance)
        X_test, y_test = lr_all.setup_and_train_model(stock, X_combined, Y_combined, tune_hyperparameters)
        all_predictions, all_probabilties = lr_all.test_model(X_test, y_test)

    for stock in trading_stocks:
        successful_execution = False
        while not successful_execution:
            trading_logs = {
                "Stock model": None,
                "Sector model": None,
                "Market model": None,
                "All model": None
            }

            """Stock trained model"""
            portfolio_stock = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
            strategy = generate_strategy(model_name, portfolio_stock, class_weight='balanced')
            lr_stock = TradingBot(portfolio_stock, strategy, initial_balance)
            X_test, y_test = lr_stock.setup_and_train_model(stock, None, None, tune_hyperparameters)
            stock_predictions, stock_probabilties = lr_stock.test_model(X_test, y_test)
            lr_stock.run_simulation(stock, stock_predictions, stock_probabilties)
            trading_logs["Stock model"] = lr_stock.get_trading_logs()

            """Sector trained model"""
            portfolio_sector = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
            strategy = generate_strategy(model_name, portfolio_sector, class_weight='balanced') #LogisticRegressionModel(portfolio_sector, class_weight='balanced')
            lr_sector = TradingBot(portfolio_sector, strategy, initial_balance)
            lr_sector.run_simulation(stock, sector_predictions, sector_probabilties)
            trading_logs["Sector model"] = lr_sector.get_trading_logs()

            """Market trained model"""
            portfolio_market = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
            strategy = generate_strategy(model_name, portfolio_market, class_weight='balanced') #LogisticRegressionModel(portfolio_market, class_weight='balanced')
            lr_market = TradingBot(portfolio_market, strategy, initial_balance)
            lr_market.run_simulation(stock, market_predictions, market_probabilties)
            trading_logs["Market model"] = lr_market.get_trading_logs()

            if trained_on_all:
                '''All trained model'''
                portfolio_all = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
                strategy = generate_strategy(model_name, portfolio_all, class_weight='balanced')# LogisticRegressionModel(portfolio_all, class_weight='balanced')
                lr_all = TradingBot(portfolio_all, strategy, initial_balance)
                lr_all.run_simulation(stock, all_predictions, all_probabilties)
                trading_logs["All model"] = lr_all.get_trading_logs()
            
            visualize_multiple_performance(
                stock_data=lr_stock.get_stock_data(),
                trade_history_logs=trading_logs,
                initial_balance=initial_balance,
                save=save_figures,
                save_dir=save_dir,
                model_name=model_name,
                hyper_opt=tune_hyperparameters,
                trained_on_all=trained_on_all
            )

            successful_execution = True
            # try:
            #     trading_logs = {
            #         "market trained model": None,
            #         "sector trained model": None,
            #         "stock trained model": None
            #     }
            #     """Stock trained model"""
            #     portfolio_stock = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
            #     strategy = LogisticRegressionModel(portfolio_stock, class_weight='balanced')
            #     lr_stock = TradingBot(portfolio_stock, strategy, initial_balance)
            #     X_test, y_test = lr_stock.setup_and_train_model(stock, None, None, True)
            #     stock_predictions, stock_probabilties = lr_stock.test_model(X_test, y_test)
            #     lr_stock.run_simulation(stock, stock_predictions, stock_probabilties)
            #     trading_logs["stock trained model"] = lr_stock.get_trading_logs()

            #     """Sector trained model"""
            #     portfolio_sector = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
            #     strategy = LogisticRegressionModel(portfolio_sector, class_weight='balanced')
            #     lr_sector = TradingBot(portfolio_sector, strategy, initial_balance)
            #     lr_sector.run_simulation(stock, sector_predictions, sector_probabilties)
            #     trading_logs["sector trained model"] = lr_sector.get_trading_logs()

            #     """Market trained model"""
            #     portfolio_market = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
            #     strategy = LogisticRegressionModel(portfolio_market, class_weight='balanced')
            #     lr_market = TradingBot(portfolio_market, strategy, initial_balance)
            #     lr_market.run_simulation(stock, market_predictions, market_probabilties)
            #     trading_logs["market trained model"] = lr_market.get_trading_logs()

            #     '''All traines model'''
            #     portfolio_all = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
            #     strategy = LogisticRegressionModel(portfolio_all, class_weight='balanced')
            #     lr_all = TradingBot(portfolio_all, strategy, initial_balance)
            #     lr_all.run_simulation(stock, all_predictions, all_probabilties)
            #     trading_logs["all trained model"] = lr_all.get_trading_logs()

            #     visualize_multiple_performance(lr_stock.get_stock_data(), trading_logs, initial_balance, False)

            #     successful_execution = True
            # except Exception as e:
            #     print(f"An error occurred with stock {stock}: {e}. Retrying...")
            #     time.sleep(60)

if __name__ == "__main__":
    main()
