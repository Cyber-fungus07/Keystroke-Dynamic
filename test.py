import os
import pandas as pd
from KNNClassifier import KeystrokeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "bio_bio.csv")
TEST_ROW = 122  # row index used for simulation

try:
    data = pd.read_csv(CSV_PATH)
    sample = data.iloc[TEST_ROW, 0:97].tolist()
    true_user_id  = str(data.iloc[TEST_ROW]["CLASS"])

    clf = KeystrokeClassifier(
        registered_biometric_file=CSV_PATH,
        typing_sample=sample,
        knn_model_test_ratio=0.3,
        neighbour_size=3,
        metric="manhattan",
        weights = 'distance',
    )

    print("EVALUATION")
    clf.knn_manhattan_holdout()
    mean_accuracy = clf.get_cv_score()
    print(f"[CV] Mean Accuracy: {mean_accuracy:.2f}%")
    best_score, best_params, best_model = clf.hyper_parameters_tuning()
    print(f"[GridSearch] Best Score: {best_score:.4f}")
    print(f"[GridSearch] Best Paraneters : {best_params}")
    print(f"[GridSearch] Best Model: {best_model}")

except Exception as e:
    print(f"Error: {e}")