from learning_models.strategy_interface import TradingStrategy
from portfolio import Portfolio
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

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

        tscv = TimeSeriesSplit(n_splits=10)

        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=tscv, scoring='f1', verbose=1)

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print(f"Best hyperparameters: {best_params}")

        # Train the final model with the best hyperparameters
        self.model = LogisticRegression(**best_params, class_weight='balanced')

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        probabilities = self.model.predict_proba(X_test)

        predictions = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        
        report = classification_report(y_test, predictions)
        print("Classification Report:\n", report)

        cm = confusion_matrix(y_test, predictions)
        print("Confusion Matrix:\n", cm)

        # Optionally, display the confusion matrix with labels
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot(cmap="Blues")  # Adjust the color map as needed
        
        return accuracy, predictions, probabilities
