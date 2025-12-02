import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer


# Import interaction_matrix so we align with its movie ordering
from interaction_matrix import interaction_matrix

movies = pd.read_csv('/Users/bencerbin/NetflixRecommendationAlgorithm/netflix-recommender/data/movies.csv')  # adjust path

"""
Returns:
    genre_features: CSR sparse matrix (num_movies x num_genres)
    genre_list: list of all genres in consistent order
    movieId_to_index: mapping for consistency checks
"""

matrixIds = interaction_matrix.columns.tolist()

movieId_to_index = {mid: i for i, mid in enumerate(matrixIds)}

genre_set = set()

for genres in movies["genres"]:
    for g in genres.split("|"):
        if g != "(no genres listed)":   #make null
            genre_set.add(g)


genre_list = sorted(list(genre_set))
num_genres = len(genre_list)
num_movies = len(matrixIds)   

genre_to_col = {g: i for i, g in enumerate(genre_list)} 

feature_matrix = lil_matrix((num_movies, num_genres), dtype=np.float32)

for _, row in movies.iterrows():
    mid = row["movieId"]

    if mid not in movieId_to_index:
        continue

    row_index = movieId_to_index[mid]

    for g in row["genres"].split("|"):
        if g in genre_to_col:
            col_index = genre_to_col[g]
            feature_matrix[row_index, col_index] = 1.0

feature_matrix = feature_matrix.tocsr()

tags = pd.read_csv('/Users/bencerbin/NetflixRecommendationAlgorithm/netflix-recommender/data/tags.csv')

def build_tfidf_matrix():
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

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        lowercase=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(documents)

    return tfidf_matrix.tocsr(), vectorizer.get_feature_names_out().tolist()





# Allow running this directly to inspect the result
# if __name__ == "__main__":
#     features, genres, mapping = build_genre_feature_matrix()
#     print("Genre feature matrix shape:", features.shape)
#     print("Genres:", genres)