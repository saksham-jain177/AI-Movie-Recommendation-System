import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Explainer:
    def __init__(self, model, features):
        self.model = model
        self.features = features
        self.scaler = MinMaxScaler()

    def explain(self, user_id, item_id, train_data):
        # Prepare input data for the model
        input_data = self._prepare_input_data(user_id, item_id, train_data)
        
        # Get the original prediction
        original_prediction = self.model.predict(input_data)[0]

        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(input_data)

        # Generate explanations
        explanations = self._generate_explanations(feature_importance)

        return original_prediction, explanations

    def _prepare_input_data(self, user_id, item_id, train_data):
        input_data = pd.DataFrame(columns=self.features)
        input_data.loc[0, 'user_id'] = user_id
        input_data.loc[0, 'item_id'] = item_id

        for feature in self.features:
            if feature.startswith('user_'):
                input_data[feature] = train_data[train_data['user_id'] == user_id][feature].iloc[0]
            elif feature.startswith('item_'):
                input_data[feature] = train_data[train_data['item_id'] == item_id][feature].iloc[0]

        return input_data

    def _calculate_feature_importance(self, input_data):
        feature_importance = []
        for feature in self.features:
            if feature not in ['user_id', 'item_id']:
                perturbed_input = input_data.copy()
                perturbed_input[feature] = np.mean(input_data[feature])
                perturbed_prediction = self.model.predict(perturbed_input)[0]
                importance = abs(self.model.predict(input_data)[0] - perturbed_prediction)
                feature_importance.append((feature, importance))

        feature_importance.sort(key=lambda x: x[1], reverse=True)
        return feature_importance

    def _generate_explanations(self, feature_importance):
        explanations = []
        importance_scores = [imp for _, imp in feature_importance]
        scaled_scores = self.scaler.fit_transform(np.array(importance_scores).reshape(-1, 1)).flatten()

        for (feature, _), score in zip(feature_importance, scaled_scores):
            if score > 0.1:  # Only include features with significant impact
                if feature.startswith('user_'):
                    explanations.append(f"Your {feature[5:]} contributed {score:.2f} to this recommendation.")
                elif feature.startswith('item_'):
                    explanations.append(f"The {feature[5:]} of this item contributed {score:.2f} to this recommendation.")
                else:
                    explanations.append(f"The {feature} contributed {score:.2f} to this recommendation.")

        return explanations