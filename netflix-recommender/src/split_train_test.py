from interaction_matrix import interaction_matrix, sparse_matrix
import numpy as np
from scipy.sparse import lil_matrix


sparse_lil = sparse_matrix.tolil()

train_matrix = lil_matrix(sparse_matrix.shape)
test_matrix = lil_matrix(sparse_matrix.shape)

np.random.seed(42)

for user_id in range(sparse_lil.shape[0]):

    rated_movies = sparse_lil.rows[user_id]

    if(len(rated_movies)) < 2:

        continue


    test_movies = np.random.choice(rated_movies, size=2, replace=False)

    for movie in rated_movies:
        rating = sparse_lil[user_id, movie]
        if movie in test_movies:
            test_matrix[user_id, movie] = rating
        else:
            train_matrix[user_id, movie] = rating
            

# Now you can use interaction_matrix or sparse_matrix directly
# print(interaction_matrix.shape)
# print(sparse_matrix.shape)


# for user_id in range(sparse_lil.shape[0]):

#     print(train_matrix.rows[user_id])


# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(8, 6))
# sns.heatmap(train_matrix[:50, :50].toarray(), cmap="viridis")
# plt.title("Train Matrix Heatmap (first 50 users/movies)")
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.heatmap(test_matrix[:50, :50].toarray(), cmap="magma")
# plt.title("Test Matrix Heatmap (first 50 users/movies)")
# plt.show()

