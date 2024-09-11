import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def calculate_metrics(y_true, y_pred, k=10):
    # Convert to numpy arrays if they're not already
    if hasattr(y_true, 'to_numpy'):
        y_true = y_true.to_numpy()
    if hasattr(y_pred, 'to_numpy'):
        y_pred = y_pred.to_numpy()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate Precision@k and Recall@k
    y_true_binary = (y_true >= 4).astype(int)  # Consider ratings >= 4 as relevant
    y_pred_top_k_indices = np.argsort(y_pred)[::-1][:k]
    
    precision_at_k = np.mean(y_true_binary[y_pred_top_k_indices])
    recall_at_k = np.sum(y_true_binary[y_pred_top_k_indices]) / np.sum(y_true_binary)
    
    # Calculate NDCG@k
    y_true_sorted = [y for _, y in sorted(zip(y_pred, y_true), key=lambda x: x[0], reverse=True)]
    ndcg_score = ndcg_at_k(y_true_sorted, k)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        f'Precision@{k}': precision_at_k,
        f'Recall@{k}': recall_at_k,
        f'NDCG@{k}': ndcg_score
    }
