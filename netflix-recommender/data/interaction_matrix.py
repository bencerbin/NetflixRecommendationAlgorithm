import pandas as pd
from scipy.sparse import csr_matrix


ratings = pd.read_csv('/Users/bencerbin/NetflixRecommendationAlgorithm/netflix-recommender/data/ratings.csv')  # adjust path


interaction_matrix = ratings.pivot(
    index='userId',      # rows = users
    columns='movieId',   # columns = movies
    values='rating'      # values = ratings
).fillna(0)             # fill missing ratings with 0


sparse_matrix = csr_matrix(interaction_matrix.values)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Convert sparse matrix to dense (only okay for small matrices)
dense_matrix = sparse_matrix.toarray()

# Optional: for large matrices, you might want to downsample
# e.g., take the first 100 users and 100 movies
dense_sample = dense_matrix[:100, :100]

# Plot using seaborn heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(dense_sample, cmap='viridis', cbar=True)
plt.xlabel("Movie ID")
plt.ylabel("User ID")
plt.title("Interaction Matrix (sample)")
plt.show()