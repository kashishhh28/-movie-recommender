import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import os
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Auto generate pkl if not exists ───────────────────
if not os.path.exists("movies.pkl") or not os.path.exists("similarity.pkl"):

    st.write("Setting up for first time... please wait ⏳")

    # Load data
    movies  = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    # Merge
    credits.columns = ["id", "title", "cast", "crew"]
    movies = movies.merge(credits, on="title")
    movies = movies[["id_x","title","overview",
                     "genres","keywords","cast_x","crew_x"]]
    movies.columns = ["movie_id","title","overview",
                      "genres","keywords","cast","crew"]

    # Convert functions
    def convert(text):
        result = []
        for item in ast.literal_eval(text):
            result.append(item["name"])
        return result

    def convert_cast(text):
        result = []
        for i, item in enumerate(ast.literal_eval(text)):
            if i == 3:
                break
            result.append(item["name"])
        return result

    def fetch_director(text):
        for item in ast.literal_eval(text):
            if item["job"] == "Director":
                return [item["name"]]
        return []

    # Apply conversions
    movies["genres"]   = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"]     = movies["cast"].apply(convert_cast)
    movies["crew"]     = movies["crew"].apply(fetch_director)

    # Remove spaces
    movies["genres"]   = movies["genres"].apply(lambda x: [i.replace(" ","") for i in x])
    movies["keywords"] = movies["keywords"].apply(lambda x: [i.replace(" ","") for i in x])
    movies["cast"]     = movies["cast"].apply(lambda x: [i.replace(" ","") for i in x])
    movies["crew"]     = movies["crew"].apply(lambda x: [i.replace(" ","") for i in x])

    # Overview to list
    movies["overview"] = movies["overview"].apply(
        lambda x: x.split() if isinstance(x, str) else [])

    # Build tags
    movies["tags"] = (movies["overview"] + movies["genres"] +
                      movies["keywords"] + movies["cast"] +
                      movies["crew"])
    movies["tags"] = movies["tags"].apply(lambda x: " ".join(x).lower())
    movies = movies[["movie_id","title","tags"]]

    # Vectorize
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["tags"]).toarray()

    # Cosine Similarity
    similarity = cosine_similarity(vectors)

    # Save pkl files
    pickle.dump(movies, open("movies.pkl", "wb"))
    pickle.dump(similarity, open("similarity.pkl", "wb"))

    st.success("Setup complete! ✅")

# ── Load saved files ───────────────────────────────────
movies     = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))
movie_list = movies["title"].values

# ── Token from environment ─────────────────────────────
token = os.getenv("BEARER_TOKEN")

# ── Fetch Poster Function ──────────────────────────────
def fetch_poster(movie_id):
    try:
        headers = {"Authorization": f"Bearer {token}"}
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        if "poster_path" not in data or data["poster_path"] is None:
            return "https://placehold.co/500x750?text=No+Poster"
        return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]
    except:
        return "https://placehold.co/500x750?text=No+Poster"

# ── Recommend Function ─────────────────────────────────
def recommend(movie):
    idx = movies[movies["title"] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[idx])),
        reverse=True,
        key=lambda x: x[1]
    )
    recommended_movies  = []
    recommended_posters = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_posters

# ── Page UI ────────────────────────────────────────────
st.title("🎬 Movie Recommendation System")
st.subheader("Find movies similar to your favourite!")

# ── Dropdown ───────────────────────────────────────────
selected_movie = st.selectbox("Select a movie:", movie_list)

# ── Show Recommendations ───────────────────────────────
if st.button("Get Recommendations 🎯"):
    names, posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])