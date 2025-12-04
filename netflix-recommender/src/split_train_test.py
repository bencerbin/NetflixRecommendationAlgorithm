# -------------------------------------------
# SPLIT TRAIN AND TEST
# -------------------------------------------
from interaction_matrix import sparse_matrix
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


# -------------------------------------------
# MAIN FUNCTION: splits interaction matrix into training data and test data based on two parameters:
#
#   min_rated -> the minimum amount of movies rated by a user to be considered
#   leave_out -> the amount of movie examples to be left out of training for testing 
#
# -------------------------------------------
def make_train_test_matrices(min_rated, leave_out):
    """
    min_rated = minimum required rated movies (5, 10, 15, ...)
    leave_out = number of movies to hold out (test size)
    """

    sparse_lil = sparse_matrix.tolil()

    train_matrix = lil_matrix(sparse_matrix.shape)
    test_matrix = lil_matrix(sparse_matrix.shape)


    """
    For every user in the matrix, either skip over it if it does not meet the threshold or 
    loop over corresponding movies and radings and assign them to the train or test matrix accordingly.
    """
    for user_id in range(sparse_lil.shape[0]):


        rated_movies = [
            movie for movie, rating in zip(sparse_lil.rows[user_id], sparse_lil.data[user_id])
            if rating > 0
        ]

        # Skip users with too few ratings
        if len(rated_movies) < min_rated:
            continue

        # Hold out N ratings
        if len(rated_movies) <= leave_out:
            continue

        test_movies = np.random.choice(rated_movies, size=leave_out, replace=False)


        for movie, rating in zip(sparse_lil.rows[user_id], sparse_lil.data[user_id]):
            if rating == 0:
                continue

            if movie in test_movies:
                test_matrix[user_id, movie] = rating
            else:
                train_matrix[user_id, movie] = rating

    return train_matrix.tocsr(), test_matrix.tocsr()
