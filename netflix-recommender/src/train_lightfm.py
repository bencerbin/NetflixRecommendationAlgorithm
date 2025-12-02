from lightfm import LightFM
from interaction_matrix import sparse_matrix
from split_train_test import train_matrix, test_matrix  # if split is in another file
from feature_matrix import feature_matrix
import numpy as np

# 1. Create the model
# For content-based filtering, use no collaborative loss (warp/cosine not needed), but we still can use warp/bpr/logistic
model = LightFM(loss='warp')  # WARP loss is popular for implicit feedback

# 2. Fit the model
# train_matrix is sparse, CSR is ideal
model.fit(train_matrix, item_features=feature_matrix, epochs=50, num_threads=4)

# 3. Make predictions
# LightFM predicts "score" for user-item pairs
user_id = 0
item_id = 1

# for item_id in range(200):

#     score = model.predict(user_ids=np.array([user_id]), item_ids=np.array([item_id]))
#     print(f"Predicted score for user {user_id}, movie {item_id}: {score}")




"""
# Get nonzero items for this user
item_ids = train_matrix[user_id].nonzero()[1]  # nonzero column indices

for item_id in item_ids:
    # Predicted score
    score = model.predict(user_ids=np.array([user_id]), item_ids=np.array([item_id]))
    
    # Actual rating from the train matrix
    actual_rating = train_matrix[user_id, item_id]  # works for CSR or dense
    
    print(f"User {user_id}, Movie {item_id}, Predicted score: {score[0]:.3f}, Actual rating: {actual_rating}")
"""

from lightfm.evaluation import precision_at_k, auc_score, recall_at_k
print("train precision: ", precision_at_k(model, train_matrix, item_features=feature_matrix, k=10).mean())
print("test precision: ", precision_at_k(model, test_matrix, item_features=feature_matrix, train_interactions=train_matrix, k=13).mean())
print("AUC:", auc_score(model, test_matrix, item_features=feature_matrix, train_interactions=train_matrix).mean())




