from learning_models.strategy_interface import TradingStrategy
from portfolio import Portfolio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

class RandomForestModel(TradingStrategy):
    def __init__(self, portfolio: Portfolio, class_weight=None):
        super().__init__(portfolio)
        self.model = RandomForestClassifier(class_weight=class_weight)

    def tune_hyperparameters(self, X_train, y_train):

        param_grid = {
            'n_estimators': [100, 200],          # Number of trees in the forest
            'max_depth': [20, None],              # Maximum depth of the tree
            'min_samples_split': [2, 10],          # Minimum samples to split an internal node
            'min_samples_leaf': [1, 4],            # Minimum samples required to be a leaf node
            'bootstrap': [True, False]                # Whether bootstrap samples are used
        }
        
        tscv = TimeSeriesSplit(n_splits=5)  # Time series split for cross-validation

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=tscv,
            scoring='accuracy',
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Get the best parameters and retrain the model with the optimal hyperparameters
        best_params = grid_search.best_params_
        print(f"Best hyperparameters: {best_params}")
        
        # Update the model with the best parameters found
        self.model = RandomForestClassifier(**best_params, class_weight='balanced')


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
        threshhold = 0.01
        quantity = int(self.portfolio.get_balance() / entry_price)
        entry_price = entry_price
        stop_loss = entry_price * (1 - threshhold / 2)
        take_profit = entry_price * (1 + threshhold)

        self.portfolio.trade_long(symbol, date, quantity, entry_price, stop_loss, take_profit)