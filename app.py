import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
import requests
import time
import random
from requests.exceptions import RequestException
from advanced_model import NeuralNetworkModel
from data_preprocessing import preprocess_data
from explainability import Explainer
import tmdbsimple as tmdb
from PIL import Image
import io

tmdb.API_KEY = "your-api-key"

def truncate_text(text, max_length=30):
    return text[:max_length] + '...' if len(text) > max_length else text

def format_movie_info(movie_info, rating):
    if movie_info:
        genres = ', '.join(movie_info['genres'][:3])
        return f"""
        **{movie_info['title']}**
        
        📅 Release: {movie_info['release_date'][:4]}\n
        🎭 Genres: {genres}\n
        ⭐ Predicted Rating: {rating:.2f}
        """
    return "Movie information unavailable"

@st.cache_data
def fetch_movie_info(movie_id, movie_dict, data_path, retries=3, delay=1):
    for attempt in range(retries):
        try:
            movie = tmdb.Movies(movie_id)
            response = movie.info()
            return {
                'title': response['title'],
                'poster_path': f"https://image.tmdb.org/t/p/w500{response['poster_path']}",
                'release_date': response['release_date'],
                'genres': [genre['name'] for genre in response['genres']]
            }
        except RequestException as e:
            if attempt == retries - 1:
                logging.warning(f"Failed to fetch info for movie {movie_id} from TMDB: {str(e)}")
                return get_local_movie_info(movie_id, movie_dict, data_path)
            time.sleep(delay)
        except Exception as e:
            logging.warning(f"Unexpected error fetching info for movie {movie_id} from TMDB: {str(e)}")
            return get_local_movie_info(movie_id, movie_dict, data_path)
    return get_local_movie_info(movie_id, movie_dict, data_path)

def get_local_movie_info(movie_id, movie_dict, data_path):
    movies_df = pd.read_csv(f"{data_path}/u.item", sep='|', encoding='latin-1', 
                            names=['movie_id', 'title', 'release_date', 'video_release_date', 
                                   'IMDb_URL'] + ['unknown'] + [f'{genre}' for genre in 
                                   ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                                    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                                    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                                    'Thriller', 'War', 'Western']])
    movie_info = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
    genres = [col for col in movies_df.columns[5:] if movie_info[col] == 1]
    return {
        'title': movie_info['title'],
        'poster_path': None,
        'release_date': movie_info['release_date'],
        'genres': genres
    }

@st.cache_resource
def load_or_train_model(features, data_path):
    model = NeuralNetworkModel(features)
    try:
        model.load_model('trained_model')
        st.success("Pre-trained model loaded successfully.")
    except (FileNotFoundError, ValueError):
        st.warning("Error loading pre-trained model. Training a new model...")
        train_data, _, _ = preprocess_data(data_path)
        X, y = train_data[features], train_data['rating']
        model.fit(X, y)
        model.save_model('trained_model')
        st.success("New model trained and saved successfully.")
    return model

@st.cache_resource
def load_explainer(_model, features):
    return Explainer(_model, features)

def explain_recommendation(model, user_id, item_id, data_path, features):
    explainer = load_explainer(model, features)
    train_data, _, _ = preprocess_data(data_path)
    return explainer.explain(user_id, item_id, train_data)

def get_top_n_recommendations(model, user_id, data_path, movie_dict, n=10):
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
    top_n = sorted(zip(items_to_predict, predictions), key=lambda x: x[1], reverse=True)
    
    return [(item, rating) for item, rating in top_n if item in movie_dict][:n]

@st.cache_data
def load_movie_data(data_path):
    movies_df = pd.read_csv(f"{data_path}/u.item", sep='|', encoding='latin-1', 
                            names=['movie_id', 'title', 'release_date', 'video_release_date', 
                                   'IMDb_URL'] + ['unknown'] + [f'{genre}' for genre in 
                                   ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                                    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                                    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                                    'Thriller', 'War', 'Western']])
    return dict(zip(movies_df.movie_id, movies_df.title))

@st.cache_data
def load_user_data(data_path):
    return pd.read_csv(f"{data_path}/u.user", sep='|', encoding='latin-1', 
                       names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])

def main():
    st.set_page_config(layout="wide", page_title="Movie Recommender")
    st.title("🎬 AI-Powered Movie Recommendation System")    

    st.markdown("""
        ## Welcome to the Movie Recommender!
        This AI-powered system suggests movies based on user preferences and behavior.
        To get started, select a user from the sidebar and click "Get Recommendations".
    """)
    
    data_path = "data/ml-100k"
    train_data, _, features = preprocess_data(data_path)
    
    with st.spinner("Loading or training the model..."):
        model = load_or_train_model(features, data_path)
    
    movie_dict = load_movie_data(data_path)
    users_df = load_user_data(data_path)

    if 'show_movie_search' not in st.session_state:
        st.session_state.show_movie_search = False

    if 'favorites' not in st.session_state:
        st.session_state.favorites = set()

    with st.sidebar:
        st.header("User Selection")
        st.markdown("Choose how you'd like to select a user:")
        search_option = st.radio("Selection method:", ("Random", "ID", "Occupation", "Age Range"), 
                                 help="Choose how you want to select a user for recommendations")
        
        if search_option == "Random":
            user_id = st.button("Get Random User", on_click=lambda: st.session_state.update({'user_id': random.choice(users_df['user_id'].tolist())}))
            user_id = st.session_state.get('user_id', random.choice(users_df['user_id'].tolist()))
        elif search_option == "ID":
            user_id = st.number_input("Enter User ID", min_value=1, max_value=users_df['user_id'].max(), value=1, step=1)
        elif search_option == "Occupation":
            occupation = st.selectbox("Select Occupation", sorted(users_df['occupation'].unique()))
            user_id = st.selectbox("Select User", users_df[users_df['occupation'] == occupation]['user_id'].tolist())
        else:  # Age Range
            age_range = st.slider("Select Age Range", min_value=int(users_df['age'].min()), max_value=int(users_df['age'].max()), value=(25, 35))
            filtered_users = users_df[(users_df['age'] >= age_range[0]) & (users_df['age'] <= age_range[1])]
            user_id = st.selectbox("Select User", filtered_users['user_id'].tolist())

        if st.button("Get Recommendations", key="recommend_button"):
            progress_bar = st.progress(0)
            with st.spinner("Generating recommendations..."):
                recommendations = get_top_n_recommendations(model, user_id, data_path, movie_dict, n=10)
                for i in range(10):
                    time.sleep(0.1)  # Short delay for visual effect
                    progress_bar.progress((i + 1) / 10)
                
                st.session_state.recommendations = recommendations
            progress_bar.empty()

        st.markdown("---")
        st.session_state.show_movie_search = st.checkbox("Show Movie Search", value=st.session_state.show_movie_search)

    if st.session_state.show_movie_search:
        st.subheader("Movie Search")
        movie_search = st.text_input("Search for a movie:")
        if movie_search:
            filtered_movies = [movie for movie in movie_dict.values() if movie_search.lower() in movie.lower()]
            if filtered_movies:
                selected_movie = st.selectbox("Select a movie:", filtered_movies)
                st.write(f"Selected movie: {selected_movie}")
            else:
                st.write("No movies found matching your search.")

    if 'recommendations' in st.session_state and st.session_state.recommendations:
        user_info = users_df[users_df['user_id'] == user_id].iloc[0]
        st.subheader(f"Top Movie Recommendations for User {user_id}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.write(f"**Age:** {user_info['age']}")
        col2.write(f"**Gender:** {user_info['gender']}")
        col3.write(f"**Occupation:** {user_info['occupation']}")
        col4.write(f"**Zip Code:** {user_info['zip_code']}")
        
        st.write("---")

        displayed_count = 0
        for i in range(0, len(st.session_state.recommendations), 4):
            row = st.columns(4)
            col_index = 0
            for j in range(4):
                if i + j < len(st.session_state.recommendations):
                    item, rating = st.session_state.recommendations[i + j]
                    movie_info = fetch_movie_info(item, movie_dict, data_path)
                    
                    if movie_info and movie_info['poster_path']:
                        with row[col_index]:
                            st.image(movie_info['poster_path'], caption=truncate_text(movie_info['title'], 20), use_column_width=True)
                            st.markdown(format_movie_info(movie_info, rating))

                            with st.expander("Explain"):
                                prediction, explanation = explain_recommendation(model, user_id, item, data_path, features)
                                st.write(f"Predicted Rating: {prediction:.2f}")
                                st.write("Top factors:")
                                for exp in explanation[:2]:
                                    st.write(f"- {exp}")
                        
                        col_index += 1
                        displayed_count += 1
                        
                        if displayed_count >= 10:
                            break
            
            if displayed_count >= 10:
                break
            
            if col_index > 0:
                st.write("---")

        if displayed_count == 0:
            st.write("No movie posters found for the recommendations.")

    if st.sidebar.checkbox("Show Favorites"):
        st.subheader("Your Favorite Movies")
        for fav_item in st.session_state.favorites:
            fav_movie_info = fetch_movie_info(fav_item, movie_dict, data_path)
            st.write(format_movie_info(fav_movie_info, 0))

if __name__ == "__main__":
    main()
