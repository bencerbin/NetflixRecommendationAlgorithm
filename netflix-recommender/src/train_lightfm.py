from lightfm import LightFM
from interaction_matrix import sparse_matrix
from split_train_test import train_matrix, test_matrix  # if split is in another file
import numpy as np

# 1. Create the model
# For content-based filtering, use no collaborative loss (warp/cosine not needed), but we still can use warp/bpr/logistic
model = LightFM(loss='warp')  # WARP loss is popular for implicit feedback

# 2. Fit the model
# train_matrix is sparse, CSR is ideal
model.fit(train_matrix, epochs=10, num_threads=4)

# 3. Make predictions
# LightFM predicts "score" for user-item pairs
user_id = 0
item_id = 1

for item_id in range(200):

    score = model.predict(user_ids=np.array([user_id]), item_ids=np.array([item_id]))
    print(f"Predicted score for user {user_id}, movie {item_id}: {score}")
