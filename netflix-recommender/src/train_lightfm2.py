# -------------------------------------------
# COLLABORATIVE + GENRES MODEL 
# -------------------------------------------

from lightfm import LightFM
from split_train_test import make_train_test_matrices
from feature_matrix import feature_matrix
import numpy as np
from lightfm.evaluation import precision_at_k, auc_score
import pandas as pd   # <-- NEW

min_rated_values = range(5, 105, 5)   # 5, 10, 15, ..., 100
leave_out_values = range(1, 21, 1)

results = []   # <-- NEW: collect results here

for leave_out in leave_out_values:

    # -------------------------------------------
    # Generate Train and Test Matrices
    # -------------------------------------------
    train_matrix, test_matrix = make_train_test_matrices(
        min_rated=leave_out*5,
        leave_out=leave_out
    )

    # -------------------------------------------
    # Train LightFM with item feature matrix
    # -------------------------------------------
    model = LightFM(loss='warp')

    model.fit(
        train_matrix,
        item_features=feature_matrix,
        epochs=50,
        num_threads=4
    )

    # -------------------------------------------
    # Evaluation
    # -------------------------------------------
    train_precision = precision_at_k(
        model, train_matrix, item_features=feature_matrix, k=5
    ).mean()

    test_precision = precision_at_k(
        model, test_matrix,
        item_features=feature_matrix,
        train_interactions=train_matrix,
        k=5
    ).mean()

    auc = auc_score(
        model, test_matrix,
        item_features=feature_matrix,
        train_interactions=train_matrix
    ).mean()

    print("train precision:", train_precision)
    print("test precision:", test_precision)
    print("AUC:", auc)

    # -------------------------------------------
    #  store results
    # -------------------------------------------
    results.append({
        "leave_out": leave_out,
        "min_rated": leave_out * 5,
        "train_precision": float(train_precision),
        "test_precision": float(test_precision),
        "auc": float(auc),
    })

# -------------------------------------------
# save results to CSV
# -------------------------------------------
df = pd.DataFrame(results)
df.to_csv("evaluation_results_genre.csv", index=False)

print("\nSaved results to evaluation_results_genre.csv")
