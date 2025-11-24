import pandas as pd
from scipy.sparse import csr_matrix


ratings = pd.read_csv('/Users/bencerbin/NetflixRecommendationAlgorithm/netflix-recommender/data/ratings.csv')  # adjust path


interaction_matrix = ratings.pivot(
    index='userId',      # rows = users
    columns='movieId',   # columns = movies
    values='rating'      # values = ratings
).fillna(0)             # fill missing ratings with 0


sparse_matrix = csr_matrix(interaction_matrix.values)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    dense_sample = sparse_matrix[:100, :100].toarray()
    plt.figure(figsize=(12, 8))
    sns.heatmap(dense_sample, cmap='viridis', cbar=True)
    plt.xlabel("Movie ID")
    plt.ylabel("User ID")
    plt.title("Interaction Matrix (sample)")
    plt.show()
