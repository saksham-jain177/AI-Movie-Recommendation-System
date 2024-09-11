import argparse
from data_preprocessing import preprocess_data
from baseline_model import BaselineModel
from advanced_model import NeuralNetworkModel, tune_model
from explainability import Explainer
import pandas as pd
import numpy as np
from evaluation import calculate_metrics
from sklearn.model_selection import KFold

def train_model(data_path, k=5):
    full_data, _, features = preprocess_data(data_path)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    baseline_metrics = []
    advanced_metrics = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(full_data), 1):
        print(f"\nFold {fold}/{k}")
        
        train_data = full_data.iloc[train_index]
        test_data = full_data.iloc[test_index]
        
        # Train and evaluate baseline model
        baseline_model = BaselineModel()
        baseline_model.fit(train_data[features], train_data['rating'])
        baseline_predictions = baseline_model.predict(test_data[features])
        baseline_fold_metrics = calculate_metrics(test_data['rating'], baseline_predictions)
        baseline_metrics.append(baseline_fold_metrics)
        
        # Tune and train advanced model
        if fold == 1:
            print("Tuning hyperparameters...")
            best_hps = tune_model(train_data[features], train_data['rating'], features, max_trials=10)
            print(f"Best hyperparameters: {best_hps.values}")
        
        advanced_model = NeuralNetworkModel(features)
        advanced_model.model = advanced_model.build_model(best_hps)
        advanced_model.fit(train_data[features], train_data['rating'])
        advanced_predictions = advanced_model.predict(test_data[features])
        advanced_fold_metrics = calculate_metrics(test_data['rating'], advanced_predictions)
        advanced_metrics.append(advanced_fold_metrics)
    
    # Calculate and print average metrics
    baseline_avg_metrics = {k: np.mean([m[k] for m in baseline_metrics]) for k in baseline_metrics[0]}
    advanced_avg_metrics = {k: np.mean([m[k] for m in advanced_metrics]) for k in advanced_metrics[0]}
    
    print("\nAverage Model Performance Comparison:")
    print("{:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Model", "RMSE", "MAE", "MSE", "Precision@10", "Recall@10", "NDCG@10"))
    print("-" * 85)
    print("{:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
        "Baseline", 
        baseline_avg_metrics['RMSE'], baseline_avg_metrics['MAE'], baseline_avg_metrics['MSE'],
        baseline_avg_metrics['Precision@10'], baseline_avg_metrics['Recall@10'], baseline_avg_metrics['NDCG@10']))
    print("{:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
        "Advanced", 
        advanced_avg_metrics['RMSE'], advanced_avg_metrics['MAE'], advanced_avg_metrics['MSE'],
        advanced_avg_metrics['Precision@10'], advanced_avg_metrics['Recall@10'], advanced_avg_metrics['NDCG@10']))

    return advanced_model, features

def explain_prediction(model, features, input_data):
    explainer = Explainer(model, features)
    prediction = model.predict(input_data)[0]
    explanation = explainer.explain(input_data['user_id'].iloc[0], input_data['item_id'].iloc[0], input_data)
    
    print(f"Predicted rating: {prediction:.2f}")
    print("\nExplanation:")
    for exp in explanation:
        print(exp)

def get_top_n_recommendations(model, user_id, data_path, n=10):
    train_data, _, features = preprocess_data(data_path)
    all_items = train_data['item_id'].unique()
    user_rated_items = train_data[train_data['user_id'] == user_id]['item_id']
    items_to_predict = list(set(all_items) - set(user_rated_items))
    
    user_data = pd.DataFrame({
        'user_id': [user_id] * len(items_to_predict),
        'item_id': items_to_predict
    })
    
    for feature in features:
        if feature not in ['user_id', 'item_id']:
            if feature.startswith('user_'):
                user_data[feature] = train_data[train_data['user_id'] == user_id][feature].iloc[0]
            elif feature.startswith('item_'):
                user_data[feature] = train_data[train_data['item_id'].isin(items_to_predict)].groupby('item_id')[feature].first().reset_index(drop=True)
    
    predictions = model.predict(user_data[features])
    top_n = sorted(zip(items_to_predict, predictions), key=lambda x: x[1], reverse=True)[:n]
    
    return top_n

def main():
    parser = argparse.ArgumentParser(description="Recommendation System")
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--recommend', type=int, help='Get recommendations for a user ID')
    parser.add_argument('--k', type=int, default=5, help='Number of folds for cross-validation')
    args = parser.parse_args()

    if args.train:
        model, features = train_model(args.data, args.k)
        model.save_model('trained_advanced_model')
    elif args.recommend is not None:
        model = NeuralNetworkModel([])  # Empty list as features will be set when loading the model
        model.load_model('trained_advanced_model')
        recommendations = get_top_n_recommendations(model, args.recommend, args.data)
        print(f"\nTop 10 recommendations for user {args.recommend}:")
        for item, rating in recommendations:
            print(f"Item {item}: Predicted rating {rating:.2f}")
        
        # Explain the top recommendation
        top_item, _ = recommendations[0]
        _, _, features = preprocess_data(args.data)
        input_data = pd.DataFrame({'user_id': [args.recommend], 'item_id': [top_item]})
        for feature in features:
            if feature not in ['user_id', 'item_id']:
                input_data[feature] = 0
        explain_prediction(model, features, input_data)
    else:
        print("Please specify either --train or --recommend <user_id>")

if __name__ == "__main__":
    main()
