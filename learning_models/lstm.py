from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from scikeras.wrappers import KerasClassifier
from sklearn.utils.class_weight import compute_class_weight
from portfolio import Portfolio
from learning_models.strategy_interface import TradingStrategy
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

class LSTMModel(TradingStrategy):
    def __init__(self, portfolio: Portfolio):
        super().__init__(portfolio)
        self.model = None
        self.trained = False  # Flag to check if the model has been trained

    def build_model(self, units=50, learning_rate=0.001):
        """Build and compile the LSTM model."""
        model = Sequential()
        model.add(Input(shape=(None, 1)))  # Define the input shape here
        model.add(LSTM(units=units, return_sequences=True))  # Input shape inferred from Input layer
        model.add(LSTM(units=units))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['AUC', 'Precision', 'Recall'])
        return model

    def tune_hyperparameters(self, X_train, y_train):
        model = KerasClassifier(build_fn=self.build_model, epochs=10, batch_size=32, verbose=0)
        
        param_grid = {
            'build_fn__units': [50, 100],  # Units for LSTM
            'batch_size': [16, 32],        # Batch size for training
        }

        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='accuracy', error_score='raise')

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print(f"Best hyperparameters: {best_params}")

        # Rebuild model with the best parameters
        self.model = self.build_model(units=best_params['build_fn__units'], learning_rate=0.001)
        self.trained = True
        
    def train_model(self, X_train, y_train):
        """Train the LSTM model with training data."""
        if isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
            y_train = y_train.values
            y_train = y_train.astype(int)

        if self.model is None:
            self.model = self.build_model()

        #smote_tomek = SMOTETomek(random_state=42)
        #X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

        # Calculate class weights if data is imbalanced
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))  # Convert to dictionary

        X_train = np.expand_dims(X_train, axis=-1)  # Reshape for LSTM input
        self.model.fit(X_train, y_train, epochs=10, batch_size=32, class_weight=class_weight_dict)
        self.trained = True

    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model on the test data and return predictions."""
        if not self.trained:
            raise ValueError("Model is not trained yet.")
        
        # Reshape the test data for LSTM input
        X_test = np.expand_dims(X_test, axis=-1)
        
        # Evaluate the model
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Extract metric values from the results
        metrics_dict = {name: value for name, value in zip(self.model.metrics_names, results)}
        auc_score = metrics_dict.get('AUC', None)

        # Print evaluation metrics
        print(f"Evaluation Results:")
        for metric_name, metric_value in metrics_dict.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        # Get predictions and generate a classification report
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)  # Convert probabilities to binary predictions
        report = classification_report(y_test, y_pred, zero_division=0)
        print("Classification Report:\n", report)
        
        return auc_score, y_pred, None

