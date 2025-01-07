from learning_models.strategy_interface import TradingStrategy
from portfolio import Portfolio
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score

class SupportVectorMachineModel(TradingStrategy):
    def __init__(self, portfolio: Portfolio, class_weight=None):
        super().__init__(portfolio)
        self.model = SVC(probability=True, class_weight=class_weight)

    def tune_hyperparameters(self, X_train, y_train):
        # Define the hyperparameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'poly', 'linear'],
            'class_weight': ['balanced', None]
        }

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        # Scoring metrics
        scoring = {
            'f1_weighted': make_scorer(f1_score, average='weighted'),
        }

        # Perform Grid Search
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=tscv,
            scoring=scoring,
            refit='f1_weighted',
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f"Best hyperparameters: {best_params}")

        # Update the model with the best parameters
        self.model = SVC(**best_params, probability=True)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate_model(self, X_test, y_test):
        probabilities = self.model.predict_proba(X_test)
        predictions = self.model.predict(X_test)

        # Accuracy and Classification Report
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0)
        print("Classification Report:\n", report)

        return accuracy, predictions, probabilities
