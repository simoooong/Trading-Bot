from abc import ABC, abstractmethod
from portfolio import Portfolio

class TradingStrategy(ABC):
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio

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

    def should_enter_trade(self, symbol, df_row):
        if self.portfolio.get_positions_symbol(symbol) is not None:
            return False

        if df_row['predictions'] == 1:
            return True
        
        return False

    def exit_trade_long(self, symbol, current_price, date):
        """Implement the logic for exiting a long trade."""
        self.portfolio.exit_long(symbol, date, current_price)

    def enter_trade_long(self, symbol, entry_price, atr, date, probability, stop_loss_multiplier=1.5, take_profit_multiplier=3.0):
        """Implement the logic for entering a long trade."""
        stop_loss = entry_price - (atr * stop_loss_multiplier)
        take_profit = entry_price + (atr * take_profit_multiplier)
        if probability is not None:
            p = probability
            q = 1 - p
            g = (take_profit - entry_price) / entry_price
            l = (entry_price - stop_loss) / entry_price
            f = (p / l) - (q / g)

            portfolio_balance = self.portfolio.get_balance()
            allocation = f * portfolio_balance
            if f > 1:
                quantity = int(self.portfolio.get_balance() / entry_price)
            elif allocation < 0:
                return
            else:
                quantity = int(allocation / entry_price)
        else:
            quantity = int(self.portfolio.get_balance() / entry_price)

        self.portfolio.trade_long(symbol, date, quantity, entry_price, stop_loss, take_profit)

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