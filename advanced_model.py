import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import keras_tuner as kt
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class NeuralNetworkModel:
    def __init__(self, features):
        self.features = features
        self.model = None
    
    def create_model(self, input_dim):
        self.model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_dim=input_dim),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        return self.model

    def fit(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        if self.model is None:
            self.model = self.build_model(kt.HyperParameters())
        self.model.fit(X, y, validation_split=validation_split, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X).flatten()

    def save_model(self, filepath):
        self.model.save(filepath + '.keras')

    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath + '.keras')

    def build_model(self, hp):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(len(self.features),)))
        
        for i in range(hp.Int('num_layers', 1, 5)):
            model.add(keras.layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                                         activation='relu'))
            model.add(keras.layers.Dropout(hp.Float(f'dropout_{i}', 0, 0.5, step=0.1)))
        
        model.add(keras.layers.Dense(1))
        
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

def tune_model(X, y, features, max_trials=10):
    def build_model(hp):
        model = NeuralNetworkModel(features)
        return model.build_model(hp)

    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=max_trials,
        directory='tuner_results',
        project_name='recommendation_system'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X, y, epochs=50, validation_split=0.2, callbacks=[stop_early])
    return tuner.get_best_hyperparameters(num_trials=1)[0]
