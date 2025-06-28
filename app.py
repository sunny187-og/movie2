import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

# ------------------ Step 1: Load and Clean Data ------------------ #
print("\U0001F4E5 Loading data...")

ratings = pd.read_csv('ratings_trimmed_3000users.csv')
credits = pd.read_csv('trimmed_credits.csv')
metadata = pd.read_csv('clean_metadata.csv', low_memory=False)
keywords = pd.read_csv('clean_keywords.csv')

ratings.dropna(inplace=True)
ratings = ratings[ratings['rating'].between(0.5, 5.0)]
metadata = metadata[metadata['id'].apply(lambda x: str(x).isdigit())]
metadata['id'] = metadata['id'].astype(int)
metadata.dropna(subset=['title', 'genres'], inplace=True)
credits.dropna(subset=['cast', 'crew'], inplace=True)
credits['id'] = credits['id'].astype(int)
keywords.dropna(subset=['keywords'], inplace=True)
keywords['id'] = keywords['id'].astype(int)

movies = metadata.merge(credits, on='id').merge(keywords, on='id')

# ------------------ Step 2: Feature Engineering ------------------ #
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
movies['soup'] = movies.apply(lambda x: f"{x['genres']} {x['keywords']} {x['top_actors']} {x['director']}", axis=1)

print("\U0001F50D Creating content-based matrix...")
vectorizer = TfidfVectorizer(stop_words='english')
soup_matrix = vectorizer.fit_transform(movies['soup'].fillna(''))
cos_sim = cosine_similarity(soup_matrix, soup_matrix)

movies = movies[['id', 'title', 'soup']].rename(columns={'id': 'movieId', 'title_x': 'title'})
movies.reset_index(drop=True, inplace=True)

# ------------------ Step 3: Collaborative Model ------------------ #
print("\U0001F527 Training collaborative model...")
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
model = SVD(n_factors=100, n_epochs=20)
model.fit(trainset)

# ------------------ Step 4: Hybrid Recommendation ------------------ #
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

def hybrid_recommend(user_id, fav_title=None, fav_actor=None, fav_director=None,
                     fav_genres=None, mood=None, topn=10):
    if user_id not in ratings['userId'].unique():
        return "❌ New user. Please use the cold-start function."

    preference_text = ''
    if fav_title:
        row = movies[movies['title'].str.lower() == fav_title.lower()]
        if not row.empty:
            preference_text += row.iloc[0]['soup'] + ' '
    if fav_genres:
        preference_text += ' '.join(fav_genres) + ' '
    if fav_actor:
        preference_text += fav_actor + ' '
    if fav_director:
        preference_text += fav_director + ' '
    if mood:
        genres = mood_genre_map.get(mood.lower().strip(), [])
        preference_text += ' '.join(genres) + ' '

    profile_vector = None
    if preference_text.strip():
        profile_vector = vectorizer.transform([preference_text])

    watched = set(ratings[ratings['userId'] == user_id]['movieId'])
    unseen = set(movies['movieId']) - watched
    predictions = [(mid, model.predict(user_id, mid).est) for mid in unseen]

    if profile_vector is not None:
        content_sim = cosine_similarity(profile_vector, soup_matrix).flatten()
        content_scores = dict(zip(movies['movieId'], content_sim))
        predictions = [(mid, 0.5 * collab + 0.5 * content_scores.get(mid, 0)) for mid, collab in predictions]

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [mid for mid, _ in predictions[:topn]]
    return movies[movies['movieId'].isin(top_movie_ids)][['movieId', 'title']]

def cold_start_hybrid_with_mood(fav_genres=None, fav_movie=None, fav_actor=None,
                                 fav_director=None, mood=None, topn=10):
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
        genres = mood_genre_map.get(mood.lower().strip(), [])
        preference_text += ' '.join(genres) + ' '

    if not preference_text.strip():
        return "Please provide at least one input."

    profile_vector = vectorizer.transform([preference_text])
    user_profiles = {}
    for uid in ratings['userId'].unique():
        liked = ratings[(ratings['userId'] == uid) & (ratings['rating'] >= 4.0)]
        liked_movies = movies[movies['movieId'].isin(liked['movieId'])]
        if liked_movies.empty: continue
        liked_vectors = soup_matrix[liked_movies.index]
        user_profiles[uid] = np.asarray(liked_vectors.mean(axis=0)).reshape(1, -1)

    similarities = {
        uid: cosine_similarity(profile_vector, profile)[0][0]
        for uid, profile in user_profiles.items()
    }

    top_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
    similar_user_ids = [uid for uid, _ in top_users]

    movie_scores = {}
    for uid in similar_user_ids:
        seen = set(ratings[ratings['userId'] == uid]['movieId'])
        for mid in movies['movieId']:
            if mid in seen: continue
            pred = model.predict(uid, mid).est
            movie_scores.setdefault(mid, []).append(pred)

    for mid in movie_scores:
        movie_scores[mid] = np.mean(movie_scores[mid])

    content_sim = cosine_similarity(profile_vector, soup_matrix).flatten()
    content_scores = dict(zip(movies['movieId'], content_sim))
    hybrid_scores = {
        mid: 0.5 * movie_scores[mid] + 0.5 * content_scores.get(mid, 0)
        for mid in movie_scores if mid in content_scores
    }

    top_movie_ids = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:topn]
    return movies[movies['movieId'].isin(top_movie_ids)][['movieId', 'title']]

# ------------------ Step 5: User Interaction ------------------ #
if __name__ == '__main__':
    user_id = input("Enter your user ID (or type 'new' if you're a new user): ").strip()

    if user_id.lower() == 'new':
        fav_movie = input("Enter a favorite movie (optional): ").strip()
        fav_actor = input("Enter your favorite actor (optional): ").strip()
        fav_director = input("Enter your favorite director (optional): ").strip()
        fav_genres = input("Enter favorite genres (comma-separated): ").split(',')
        mood = input("Enter your current mood (optional): ").strip()

        result = cold_start_hybrid_with_mood(
            fav_genres=[g.strip() for g in fav_genres if g.strip()],
            fav_movie=fav_movie,
            fav_actor=fav_actor,
            fav_director=fav_director,
            mood=mood
        )
    else:
        try:
            user_id = int(user_id)
        except ValueError:
            print("❌ Invalid user ID.")
            exit()

        if user_id in ratings['userId'].unique():
            fav_movie = input("Enter a favorite movie (optional): ").strip()
            fav_actor = input("Enter your favorite actor (optional): ").strip()
            fav_director = input("Enter your favorite director (optional): ").strip()
            fav_genres = input("Enter favorite genres (comma-separated): ").split(',')
            mood = input("Enter your current mood (optional): ").strip()

            result = hybrid_recommend(
                user_id=user_id,
                fav_title=fav_movie,
                fav_actor=fav_actor,
                fav_director=fav_director,
                fav_genres=[g.strip() for g in fav_genres if g.strip()],
                mood=mood
            )
        else:
            print("⚠️ User ID not found in existing users. Treating as new user...")
            fav_movie = input("Enter a favorite movie (optional): ").strip()
            fav_actor = input("Enter your favorite actor (optional): ").strip()
            fav_director = input("Enter your favorite director (optional): ").strip()
            fav_genres = input("Enter favorite genres (comma-separated): ").split(',')
            mood = input("Enter your current mood (optional): ").strip()

            result = cold_start_hybrid_with_mood(
                fav_genres=[g.strip() for g in fav_genres if g.strip()],
                fav_movie=fav_movie,
                fav_actor=fav_actor,
                fav_director=fav_director,
                mood=mood
            )

    print("\n\U0001F3AF Recommended Movies:\n", result)

