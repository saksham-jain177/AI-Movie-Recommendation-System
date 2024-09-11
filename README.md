# AI-Powered Movie Recommendation System

## Project Overview

This project implements an advanced, AI-powered movie recommendation system using collaborative filtering and neural networks. It provides personalized movie suggestions based on user preferences and behavior, along with explanations for the recommendations.

## Features

- User-based and item-based collaborative filtering
- Neural network model for advanced predictions
- Explainable AI (XAI) for recommendation insights
- Interactive web interface using Streamlit
- Cross-validation and hyperparameter tuning
- Comprehensive evaluation metrics

## Dataset

This project uses the MovieLens 100K dataset, which contains 100,000 ratings from 943 users on 1,682 movies. The dataset is widely used in the recommender system research community and provides a good balance between size and computational requirements.

Dataset characteristics:
- 100,000 ratings (1-5) from 943 users on 1,682 movies
- Each user has rated at least 20 movies
- Simple demographic info for the users (age, gender, occupation, zip)
- Movie information including title, release date, and genre

You can download the dataset [here](https://grouplens.org/datasets/movielens/100k/).

## File Descriptions

- `app.py`: Main Streamlit application file for the web interface
- `main.py`: Command-line interface for training and testing the model
- `data_preprocessing.py`: Functions for loading and preprocessing the dataset
- `baseline_model.py`: Implementation of a simple baseline recommendation model
- `advanced_model.py`: Neural network-based recommendation model with hyperparameter tuning
- `evaluation.py`: Functions for calculating various evaluation metrics
- `explainability.py`: Implementation of the explanation system for recommendations

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/saksham-jain177/AI-Movie-Recommendation-System.git
   cd AI-Movie-Recommendation-System
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Web Interface

To run the Streamlit web application:
``` streamlit run app.py ```

This will launch the web interface where you can interact with the recommendation system.

### Command-line Interface

To train the model:
``` python main.py --data path/to/dataset --train ```

To get recommendations for a specific user:
``` python main.py --data path/to/dataset --recommend user_id ```

## Dependencies

The main dependencies for this project are:

- Python 3.7+
- TensorFlow 2.x
- Keras
- Streamlit
- Pandas
- NumPy
- Scikit-learn

A complete list of dependencies can be found in the `requirements.txt` file.

## Future Improvements

- Implement content-based filtering using movie features
- Integrate more recent and larger datasets
- Develop a hybrid recommendation system combining collaborative and content-based approaches
- Implement user feedback and continuous learning
- Enhance the UI with more interactive features and visualizations

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
