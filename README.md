# IMDB_Movie_Recommendation_System_Using_Storylines

A content-based movie recommendation system built using IMDb 2024 movie data, Machine Learning, and Streamlit.
The system recommends movies based on user-entered keywords or descriptions, even when written in the user’s own words.

# Project Overview

This project scrapes IMDb movie data for films released in 2024 and builds a movie recommendation engine using Natural Language Processing (NLP).

Users can:

  - Enter a movie name or describe a movie in their own words

  - Choose how many recommendations they want

  - Get the most relevant movie suggestions instantly via a web interface

# Features

1. Web scraping of IMDb 2024 movies using Selenium

2. Cleaned and processed dataset (~10,000 movies)

3. Content-based recommendation using TF-IDF & Cosine Similarity

4. Interactive Streamlit web application

5. User-controlled number of recommendations

6. Fast performance using Streamlit caching


# Tech Stack

 - Python

 - Selenium – Web scraping

 - Pandas & NumPy – Data processing

 - Scikit-learn – TF-IDF & similarity computation

 - Streamlit – Web application

 - IMDb – Data source


# Project Structure
├── imdb_scraper.py              # IMDb web scraping script
├── imdb_2024_movies.csv         # Scraped movie dataset
├── app.py                       # Streamlit application
├── README.md                    # Project documentation


# Dataset Details

Source: IMDb Advanced Movie Search

Year: 2024

Total Movies: ~10,000

Columns:

   - Movie Title

   - Storyline


# How the Recommendation System Works

Data Preparation

  - Movie title and storyline are combined into a single text feature.

  - Duplicate records are removed.

Text Vectorization

  - TF-IDF Vectorizer converts text into numerical vectors.

  - Uses unigrams and bigrams with English stopwords removed.

Similarity Calculation

  - Cosine similarity measures how close the user query is to each movie.

Recommendation

  - Top-N most similar movies are returned based on user preference.


# Streamlit Application
User Inputs:

  - Movie name or custom description

  - Number of recommendations (slider)

Output:

  - List of recommended movies with storylines


# How to Run the Project
Install Dependencies

    pip install streamlit pandas scikit-learn selenium

Run the Streamlit App
    streamlit run app.py

