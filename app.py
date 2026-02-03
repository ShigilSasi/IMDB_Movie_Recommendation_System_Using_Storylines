import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Clean duplicates
    df.drop_duplicates(inplace=True)
    df['Combined_movie_storyline'] = df['Movie Title'] + " " + df['Storyline']
    return df

# -------------------------------
# TF-IDF VECTOR
# -------------------------------
@st.cache_data
def create_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=1500, ngram_range=(1,2))
    tfidf_matrix = tfidf.fit_transform(df['Combined_movie_storyline'])
    return tfidf, tfidf_matrix

# -------------------------------
# RECOMMENDATION FUNCTION
# -------------------------------
def search_movie(query, df, tfidf_obj, tfidf_matrix, k=5):
    query = query.lower()
    query_vec = tfidf_obj.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity.argsort()[-k:][::-1]
    return df.iloc[top_indices][['Movie Title', 'Storyline']]

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="IMDb 2024 Movie Recommender", layout="wide")
st.title("🎬 IMDb 2024 Movie Recommender")

# Load data
df = load_data("imdb_2024_movies.csv")
tfidf_obj, tfidf_matrix = create_tfidf_matrix(df)

# Input query
query = st.text_input("Enter a movie name or keyword from storyline:")

# Slider for number of recommendations
num_results = st.slider("How many movie recommendations do you want?", min_value=1, max_value=20, value=5)

if query:
    with st.spinner("Searching for movies..."):
        results = search_movie(query, df, tfidf_obj, tfidf_matrix, k=num_results)
    
    st.subheader(f"Top {num_results} Recommendations")
    for idx, row in results.iterrows():
        st.markdown(f"**{row['Movie Title']}**")
        st.write(row['Storyline'])
        st.markdown("---")
else:
    st.info("Type a movie name or a keyword to get recommendations")
