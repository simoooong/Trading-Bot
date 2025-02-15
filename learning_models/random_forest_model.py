from learning_models.strategy_interface import TradingStrategy
from portfolio import Portfolio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score, confusion_matrix, ConfusionMatrixDisplay

class RandomForestModel(TradingStrategy):
    def __init__(self, portfolio: Portfolio, class_weight):
        super().__init__(portfolio)
        self.model = RandomForestClassifier(class_weight=class_weight)

    def tune_hyperparameters(self, X_train, y_train):

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [20, None],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 4],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample']
        }

        tscv = TimeSeriesSplit(n_splits=5)

        scoring = {
            'f1_weighted': make_scorer(f1_score, average='weighted'),
        }

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

        self.model = RandomForestClassifier(**best_params)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate_model(self, X_test, y_test):
        probabilities = self.model.predict_proba(X_test)
        predictions = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        report = classification_report(y_test, predictions, zero_division=0)
        print("Classification Report:\n", report)

        cm = confusion_matrix(y_test, predictions)
        print("Confusion Matrix:\n", cm)

        # Optionally, display the confusion matrix with labels
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot(cmap="Blues")  # Adjust the color map as needed

        return accuracy, predictions, probabilities
