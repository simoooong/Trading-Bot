from data.api_handler import ApiClient
from data.sqlite_database import SQLiteDatabase
from data.stock_data_service import StockDataService
from data.preprocessing_data import PreprocessData
from portfolio import Portfolio
from trading_system import TradingSystem
from learning_models.strategy_interface import TradingStrategy
from learning_models.logistic_regression_model import LogisticRegressionModel
from learning_models.random_forest_model import RandomForestModel
from visualize_trading_results import visualize_performance 


class TradingBot:
    def __init__(self, portfolio: Portfolio, strategy: TradingStrategy, initial_balance=100000):
        # Initialize the portfolio and service
        self.api_client = ApiClient(interval="60min")
        self.data_base = SQLiteDatabase("persistence/stock_data.db")
        self.data_service = StockDataService(self.data_base, self.api_client)
        self.preprocessor = PreprocessData(self.data_base, self.data_service)
        
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

def main():
    initial_balance = 100000
    trading_stocks = ["JNJ", "AAPL", "XLV", "SPY"]

    '''
    Model: Logistic Regression
    Stock: JNJ
    '''
    portfolio_lr_jnj = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
    strategy = LogisticRegressionModel(portfolio_lr_jnj, class_weight='balanced')
    lr_jnj = TradingBot(portfolio_lr_jnj, strategy, initial_balance)

    X_test, y_test = lr_jnj.setup_and_train_model(trading_stocks[3])
    predictions = lr_jnj.test_model(X_test, y_test)
    lr_jnj.run_simulation(trading_stocks[0], predictions)
    lr_jnj.show_results()
    lr_jnj.show_trading_logs()
    lr_jnj.visualize_performance()

    # '''
    # Model: Logistic Regression
    # Stock: AAPL
    # '''
    # portfolio_lr_aapl =  Portfolio(risk_tolerance=1, initial_balance=initial_balance)
    # strategy = LogisticRegressionModel(portfolio_lr_aapl, class_weight='balanced')
    # lr_aapl = TradingBot(portfolio_lr_aapl, strategy, initial_balance)

    # X_test, y_test = lr_aapl.setup_and_train_model(trading_stocks[1])
    # predictions = lr_aapl.test_model(X_test, y_test)
    # lr_aapl.run_simulation(trading_stocks[1], predictions)
    # lr_aapl.show_results()
    # lr_aapl.visualize_performance()
    
    # '''
    # Model: Random Forest
    # Stock: JNJ
    # '''
    # portfolio_rf_jnj =  Portfolio(risk_tolerance=1, initial_balance=initial_balance)
    # strategy = RandomForestModel(portfolio_rf_jnj, class_weight='balanced')
    # rf_aapl = TradingBot(portfolio_rf_jnj, strategy, initial_balance)

    # X_test, y_test = rf_aapl.setup_and_train_model(trading_stocks[0])
    # predictions = rf_aapl.test_model(X_test, y_test)
    # rf_aapl.run_simulation(trading_stocks[0], predictions)
    # rf_aapl.show_results()
    
    # '''
    # Model: Random Forest
    # Stock: AAPL
    # '''
    # portfolio_rf_aapl =  Portfolio(risk_tolerance=1, initial_balance=initial_balance)
    # strategy = RandomForestModel(portfolio_rf_aapl, class_weight='balanced')
    # rf_aapl = TradingBot(portfolio_rf_aapl, strategy, initial_balance)

    # X_test, y_test = rf_aapl.setup_and_train_model(trading_stocks[1])
    # predictions = rf_aapl.test_model(X_test, y_test)
    # rf_aapl.run_simulation(trading_stocks[1], predictions)
    # rf_aapl.show_results()

if __name__ == "__main__":
    main()
