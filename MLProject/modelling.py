"""
modelling.py untuk MLflow Project (dijalankan oleh workflow CI).
"""
import argparse
import os
import sys
import urllib.error
import urllib.request

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET = "Credit_Score"


def parse_args():
    parser = argparse.ArgumentParser(description="Latih model credit scoring dengan MLflow autolog")
    parser.add_argument(
        "--data_dir",
        default=os.path.join(BASE_DIR, "creditscore_preprocessing"),
        help="Folder berisi train_preprocessed.csv dan test_preprocessed.csv",
    )
    parser.add_argument(
        "--tracking_uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="Alamat MLflow Tracking. Kosong = penyimpanan lokal ./mlruns",
    )
    parser.add_argument("--experiment_name", default="Credit Scoring - Fadli Nur Rofik")
    return parser.parse_args()


def server_reachable(uri, timeout=3):
    """Cek cepat apakah MLflow Tracking Server (http) menjawab."""
    try:
        urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except OSError:
        return False


def load_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train_preprocessed.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test_preprocessed.csv"))

    X_train = train.drop(columns=[TARGET]).astype("float64")
    X_test = test.drop(columns=[TARGET]).astype("float64")
    y_train = train[TARGET].astype(int)
    y_test = test[TARGET].astype(int)
    return X_train, X_test, y_train, y_test


def main():
    args = parse_args()
    under_project = "MLFLOW_RUN_ID" in os.environ  # True bila dijalankan oleh `mlflow run`

    if args.tracking_uri:
        if args.tracking_uri.startswith("http") and not server_reachable(args.tracking_uri):
            sys.exit(f"MLflow Tracking Server tidak dapat dijangkau di {args.tracking_uri}")
        mlflow.set_tracking_uri(args.tracking_uri)

    if not under_project:
        mlflow.set_experiment(args.experiment_name)

    X_train, X_test, y_train, y_test = load_data(args.data_dir)
    print(f"Data latih: {X_train.shape} | Data uji: {X_test.shape}")

    mlflow.sklearn.autolog(log_input_examples=True)

    with mlflow.start_run(run_name=None if under_project else "random_forest_ci"):
        # Nilai hyperparameter tetap (tanpa tuning), sama dengan Kriteria 2
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        mlflow.log_metrics({"test_accuracy": accuracy, "test_f1_macro": f1_macro})

        print(f"Akurasi data uji : {accuracy:.4f}")
        print(f"F1-macro data uji: {f1_macro:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Poor", "Standard", "Good"]))


if __name__ == "__main__":
    main()
