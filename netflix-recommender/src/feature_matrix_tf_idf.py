import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

# Import interaction_matrix to keep movieId ordering consistent
from interaction_matrix import interaction_matrix

# Load raw data
movies = pd.read_csv('/Users/bencerbin/NetflixRecommendationAlgorithm/netflix-recommender/data/movies.csv')
tags = pd.read_csv('/Users/bencerbin/NetflixRecommendationAlgorithm/netflix-recommender/data/tags.csv')

# MovieId ordering from interaction_matrix
movieIds = interaction_matrix.columns.tolist()
movieId_to_index = {mid: i for i, mid in enumerate(movieIds)}

# ----------------------------------------------------------
# 1. GENRE FEATURES
# ----------------------------------------------------------

def build_genre_feature_matrix(movieId_to_index, movies):
    """
    Builds a genre-based item feature matrix (num_movies x num_genres).
    """

    # Build genre vocabulary
    genre_set = set()
    for genres in movies["genres"]:
        for g in genres.split("|"):
            if g != "(no genres listed)":
                genre_set.add(g)

    genre_list = sorted(list(genre_set))
    num_genres = len(genre_list)
    num_movies = len(movieId_to_index)

    genre_to_col = {g: i for i, g in enumerate(genre_list)}

    # Allocate matrix
    genre_matrix = lil_matrix((num_movies, num_genres), dtype=np.float32)

    # Fill the matrix
    for _, row in movies.iterrows():
        mid = row["movieId"]

        if mid not in movieId_to_index:
            continue

        row_idx = movieId_to_index[mid]

        for g in row["genres"].split("|"):
            if g in genre_to_col:
                col_idx = genre_to_col[g]
                genre_matrix[row_idx, col_idx] = 1.0

    return genre_matrix.tocsr(), genre_list


# ----------------------------------------------------------
# 2. TF-IDF TAG FEATURES
# ----------------------------------------------------------

def build_tfidf_tag_matrix(movieId_to_index, tags):
    """
    Builds a TF-IDF matrix for movie tags (num_movies x vocab_size).
    Each movie gets one text document compiled from all its tags.
    """

    num_movies = len(movieId_to_index)

    # Group tags into one document per movie
    tag_docs_series = tags.groupby("movieId")["tag"].apply(
        lambda x: " ".join(str(t) for t in x)
    )
    tag_docs = tag_docs_series.to_dict()

    # Build documents aligned with interaction_matrix movie order
    documents = []
    for mid in sorted(movieId_to_index, key=lambda m: movieId_to_index[m]):
        if mid in tag_docs:
            documents.append(tag_docs[mid])
        else:
            documents.append("")

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,  # caps dimensionality for performance
        stop_words="english",
        lowercase=True
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    return tfidf_matrix.tocsr(), vectorizer.get_feature_names_out().tolist()


# ----------------------------------------------------------
# 3. MAIN: BUILD FINAL ITEM_FEATURES MATRIX
# ----------------------------------------------------------

# Build content features
genre_matrix, genre_list = build_genre_feature_matrix(movieId_to_index, movies)
tfidf_matrix, tfidf_vocab = build_tfidf_tag_matrix(movieId_to_index, tags)

# Combine genre + TF-IDF side-by-side
item_features = hstack([genre_matrix, tfidf_matrix]).tocsr()

# Exported publicly
__all__ = ["item_features", "genre_matrix", "tfidf_matrix", "genre_list", "tfidf_vocab"]


# ----------------------------------------------------------
# For debugging / exploration
# ----------------------------------------------------------

if __name__ == "__main__":
    print("Genre matrix shape:", genre_matrix.shape)
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    print("Combined item_features shape:", item_features.shape)
    print("Number of genres:", len(genre_list))
    print("TF-IDF vocab size:", len(tfidf_vocab))
