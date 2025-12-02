from interaction_matrix import interaction_matrix, sparse_matrix
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

# Convert original sparse matrix to LIL for easy row access
sparse_lil = sparse_matrix.tolil()

# Initialize empty train/test matrices
train_matrix = lil_matrix(sparse_matrix.shape)
test_matrix = lil_matrix(sparse_matrix.shape)

np.random.seed(33)

for user_id in range(sparse_lil.shape[0]):

    # Movies the user actually rated (rating > 0)
    rated_movies = [
        movie for movie, rating in zip(sparse_lil.rows[user_id], sparse_lil.data[user_id])
        if rating > 0
    ]

    leaveOut = 5

    # Need at least 2 to split
    if len(rated_movies) < leaveOut*5:
        continue

    # Randomly select 2 test movies
    test_movies = np.random.choice(rated_movies, size=leaveOut, replace=False)

    # Fill train and test matrices
    for movie, rating in zip(sparse_lil.rows[user_id], sparse_lil.data[user_id]):
        if rating == 0:
            continue  # skip non-interactions

        if movie in test_movies:
            test_matrix[user_id, movie] = rating
        else:
            train_matrix[user_id, movie] = rating

# Convert to CSR (required by LightFM)
train_matrix = train_matrix.tocsr()
test_matrix = test_matrix.tocsr()

# Optional debug prints
# print(train_matrix.shape, test_matrix.shape)


