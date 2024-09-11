import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(data_path):
    # Load user-item interactions
    interactions = pd.read_csv(f'{data_path}/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

    # Load movie metadata
    items = pd.read_csv(f'{data_path}/u.item', sep='|', encoding='ISO-8859-1', header=None, 
                        names=['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 
                               'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                               'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

    # Load user metadata
    users = pd.read_csv(f'{data_path}/u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])

    # Merge interactions with item (movie) metadata
    interactions_items = pd.merge(interactions, items[['item_id', 'title']], on='item_id')

    # Merge the result with user metadata
    full_data = pd.merge(interactions_items, users[['user_id', 'age', 'gender', 'occupation']], on='user_id')

    # Convert user_id and item_id into numerical values
    full_data['user_id'] = full_data['user_id'].astype('category').cat.codes
    full_data['item_id'] = full_data['item_id'].astype('category').cat.codes

    # Convert gender to numerical (0 = female, 1 = male)
    full_data['gender'] = full_data['gender'].apply(lambda x: 1 if x == 'M' else 0)

    # Feature Engineering
    user_avg_rating = full_data.groupby('user_id')['rating'].mean().reset_index(name='user_avg_rating')
    item_avg_rating = full_data.groupby('item_id')['rating'].mean().reset_index(name='item_avg_rating')
    user_rating_count = full_data.groupby('user_id')['rating'].count().reset_index(name='user_rating_count')
    item_rating_count = full_data.groupby('item_id')['rating'].count().reset_index(name='item_rating_count')

    full_data = pd.merge(full_data, user_avg_rating, on='user_id')
    full_data = pd.merge(full_data, item_avg_rating, on='item_id')
    full_data = pd.merge(full_data, user_rating_count, on='user_id')
    full_data = pd.merge(full_data, item_rating_count, on='item_id')

    # Select features for training
    data = full_data[['user_id', 'item_id', 'rating', 'user_avg_rating', 'item_avg_rating', 'user_rating_count', 'item_rating_count']]

    # Split into training (80%) and testing (20%) sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    features = ['user_id', 'item_id', 'user_avg_rating', 'item_avg_rating', 'user_rating_count', 'item_rating_count']
    return train_data, test_data, features
