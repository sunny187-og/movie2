import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# ---------------------- Load Data ----------------------
@st.cache_data
def load_data():
    movies = pd.read_csv('clean_metadata.csv')
    ratings = pd.read_csv('ratings_3000users.csv')
    credits = pd.read_csv('trimmed_credits.csv')
    keywords = pd.read_csv('clean_keywords.csv')

    # Clean and merge
    movies = movies[movies['id'].apply(lambda x: str(x).isdigit())]
    movies['id'] = movies['id'].astype(int)
    credits['id'] = credits['id'].astype(int)
    keywords['id'] = keywords['id'].astype(int)
    movies = movies.merge(credits, on='id').merge(keywords, on='id')
    movies = movies.head(5000)  # Limit to top 5000

    return movies, ratings

movies, ratings = load_data()

# ---------------------- Feature Extraction ----------------------
def extract_names(json_str, key='name', topn=None):
    try:
        items = ast.literal_eval(json_str)
        names = [item[key] for item in items]
        return ' '.join(names[:topn]) if topn else ' '.join(names)
    except:
        return ''

def get_director(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
        for member in crew:
            if member['job'] == 'Director':
                return member['name']
        return ''
    except:
        return ''

movies['genres'] = movies['genres'].apply(extract_names)
movies['keywords'] = movies['keywords'].apply(extract_names)
movies['top_actors'] = movies['cast'].apply(lambda x: extract_names(x, topn=3))
movies['director'] = movies['crew'].apply(get_director)
movies['soup'] = movies.apply(lambda row: f"{row['genres']} {row['keywords']} {row['top_actors']} {row['director']}", axis=1)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
soup_matrix = tfidf.fit_transform(movies['soup'].fillna(''))
cos_sim = cosine_similarity(soup_matrix, soup_matrix)
movies = movies[['id', 'title', 'soup']].rename(columns={'id': 'movieId'})
movies.reset_index(drop=True, inplace=True)

# ---------------------- Recommenders ----------------------
def get_content_recs(title, topn=10):
    idx = movies[movies['title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return pd.DataFrame({'title': [f"Movie '{title}' not found."]})
    idx = idx[0]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:topn+1]
    return movies.iloc[[i[0] for i in sim_scores]][['movieId', 'title']]

def cold_start_recommendation(fav_genres=None, fav_movie=None, fav_actor=None, fav_director=None, mood=None, topn=10):
    mood_genre_map = {
        'happy': ['Comedy', 'Family', 'Adventure'],
        'sad': ['Drama', 'Romance'],
        'angry': ['Action', 'Thriller'],
        'bored': ['Fantasy', 'Sci-Fi', 'Mystery'],
        'romantic': ['Romance', 'Drama'],
        'excited': ['Action', 'Adventure'],
        'scared': ['Horror', 'Thriller'],
        'inspired': ['Biography', 'History', 'Documentary']
    }

    preference_text = ''
    if fav_genres:
        preference_text += ' '.join(fav_genres) + ' '
    if fav_movie:
        movie_row = movies[movies['title'].str.lower() == fav_movie.lower()]
        if not movie_row.empty:
            preference_text += movie_row.iloc[0]['soup'] + ' '
    if fav_actor:
        preference_text += fav_actor + ' '
    if fav_director:
        preference_text += fav_director + ' '
    if mood:
        mood = mood.lower().strip()
        mood_genres = mood_genre_map.get(mood, [])
        preference_text += ' '.join(mood_genres) + ' '

    if not preference_text.strip():
        return pd.DataFrame({'title': ['Please enter some preferences.']})

    profile_vector = tfidf.transform([preference_text])
    similarity_scores = cosine_similarity(profile_vector, soup_matrix).flatten()
    top_indices = similarity_scores.argsort()[-topn:][::-1]
    return movies.iloc[top_indices][['movieId', 'title']]

def get_collab_recs(user_id, topn=10):
    return pd.DataFrame({'title': ["Collaborative filtering disabled on Streamlit Cloud."]})

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")

user_type = st.radio("Are you an existing user?", ["Yes", "No"])

if user_type == "Yes":
    user_id = st.number_input("Enter your user ID (1–3000):", min_value=1, step=1)
    fav_movie = st.text_input("What's a movie you liked?")
    mood = st.text_input("What's your current mood? (optional)")
    if st.button("Get Recommendations"):
        st.subheader("📌 Content-Based Recommendations")
        st.dataframe(get_content_recs(fav_movie))
        st.subheader("🤝 Collaborative Recommendations")
        st.dataframe(get_collab_recs(user_id))
elif user_type == "No":
    fav_movie = st.text_input("Name a movie you liked (optional)")
    fav_genres = st.text_input("Favorite genres (comma-separated)").split(',')
    fav_actor = st.text_input("Favorite actor (optional)")
    fav_director = st.text_input("Favorite director (optional)")
    mood = st.text_input("What's your current mood? (optional)")
    if st.button("Recommend Movies"):
        st.subheader("🌟 Cold Start Hybrid Recommendations")
        st.dataframe(cold_start_recommendation(fav_genres, fav_movie, fav_actor, fav_director, mood))

