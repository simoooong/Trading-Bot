from learning_models.strategy_interface import TradingStrategy
from portfolio import Portfolio
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

class LogisticRegressionModel(TradingStrategy):
    def __init__(self, portfolio: Portfolio, class_weight=None):
        super().__init__(portfolio)
        self.model = LogisticRegression(class_weight=class_weight, max_iter=5000)

    def tune_hyperparameters(self, X_train, y_train):
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10],  # Regularization strength
            #'penalty': ['l1', 'l2']       # L1 or L2 regularization
            'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky'] # Different solvers for Logistic Regression
        }

        tscv = TimeSeriesSplit(n_splits=20)

        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=tscv, scoring='accuracy', verbose=1)

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print(f"Best hyperparameters: {best_params}")

        # Train the final model with the best hyperparameters
        self.model = LogisticRegression(**best_params, class_weight='balanced')

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
    
    def enter_trade_long(self, symbol, entry_price, atr,  date,  stop_loss_multiplier=1.5, take_profit_multiplier=3.0):
        stop_loss = entry_price - (atr * stop_loss_multiplier)
        take_profit = entry_price + (atr * take_profit_multiplier)
        quantity = int(self.portfolio.get_balance() / entry_price)

        self.portfolio.trade_long(symbol, date, quantity, entry_price, stop_loss, take_profit)