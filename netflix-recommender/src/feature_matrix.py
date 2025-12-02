import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

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




# Allow running this directly to inspect the result
if __name__ == "__main__":
    features, genres, mapping = build_genre_feature_matrix()
    print("Genre feature matrix shape:", features.shape)
    print("Genres:", genres)