from abc import ABC, abstractmethod
from portfolio import Portfolio

class TradingStrategy(ABC):
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio

    @abstractmethod
    def should_enter_trade(self, symbol, price_data):
        """Determine if the strategy should enter a trade."""
        pass

    @abstractmethod
    def enter_trade_long(self, symbol, entry_price, date):
        """Implement the logic for entering a long trade."""
        pass

    @abstractmethod
    def tune_hyperparameters(self, X_train, y_train):
        '''
        Tune the hyperparameters of the model using the provided training data (X_train, y_train).
        This function will implement strategies like GridSearchCV or RandomizedSearchCV to find the optimal
        hyperparameters for the model.
        '''
        pass

    @abstractmethod
    def train_model(self, X_train_y_train):
        """
        Train the machine learning model using the provided training data and labels.
        Implementations of this method should define how the model learns from the data 
        to make predictions.
        """
        pass

    @abstractmethod
    def evaluate_model(X_test, y_test):
        """
        Evaluate the trained model on the provided test data and labels.
        Implementations should calculate performance metrics (e.g., accuracy) 
        and return predictions based on the test data.
        """
        pass


    def exit_trade_long(self, symbol, current_price, date):
        """Implement the logic for exiting a long trade."""
        self.portfolio.exit_long(symbol, date, current_price)

    def check_for_exit(self, symbol, current_price):
        """Check if the trade should be exited based on strategy rules."""
        positions = self.portfolio.get_positions()
        
        if symbol not in positions:
            return False
    
        position = positions[symbol]
        
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
    
        if current_price <= stop_loss or current_price >= take_profit:
            return True
    
        return False