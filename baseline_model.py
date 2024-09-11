import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class BaselineModel:
    def __init__(self):
        self.global_average = None
        self.user_biases = None
        self.item_biases = None

    def fit(self, X, y):
        self.global_average = y.mean()
        data = X.copy()
        data['rating'] = y
        self.user_biases = data.groupby('user_id')['rating'].mean() - self.global_average
        self.item_biases = data.groupby('item_id')['rating'].mean() - self.global_average

    def predict(self, X):
        predictions = pd.Series(self.global_average, index=X.index)
        predictions += X['user_id'].map(self.user_biases).fillna(0)
        predictions += X['item_id'].map(self.item_biases).fillna(0)
        return predictions.clip(1, 5)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MSE': mse
        }
