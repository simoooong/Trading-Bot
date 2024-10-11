from learning_models.strategy_interface import TradingStrategy
from portfolio import Portfolio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class LogisticRegressionModel(TradingStrategy):
    def __init__(self, portfolio: Portfolio, class_weight=None):
        super().__init__(portfolio)
        self.model = LogisticRegression(class_weight=class_weight)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate_model(self, X_test, y_test):
        # Get continuous predictions from the linear regression model
        predictions = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        
        report = classification_report(y_test, predictions)
        print("Classification Report:\n", report)
        
        return accuracy, predictions

    def should_enter_trade(self, symbol, df_row):
        if self.portfolio.get_positions_symbol(symbol) is not None:
            return False

        if df_row['predictions'] == 1:
            return True
        
        return False
    
    def enter_trade_long(self, symbol, entry_price, date):
        quantity = int(self.portfolio.get_balance() / entry_price)
        entry_price = entry_price
        stop_loss = entry_price * (1- 0.0121 / 2)
        take_profit = entry_price * 1.0121

        self.portfolio.trade_long(symbol, date, quantity, entry_price, stop_loss, take_profit)