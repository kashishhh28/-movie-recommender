import streamlit as st
import pickle
import pandas as pd
import requests
import os
from dotenv import load_dotenv

# ── Load .env file ────────────────────────────────────
load_dotenv()
token = os.getenv("BEARER_TOKEN")

# ── Load saved files ──────────────────────────────────
movies     = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))
movie_list = movies["title"].values

# ── Fetch Poster Function ─────────────────────────────
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

# ── Recommend Function ────────────────────────────────
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

# ── Page Title ────────────────────────────────────────
st.title("🎬 Movie Recommendation System")
st.subheader("Find movies similar to your favourite!")

# ── Dropdown ──────────────────────────────────────────
selected_movie = st.selectbox("Select a movie:", movie_list)

# ── Show Recommendations ──────────────────────────────
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