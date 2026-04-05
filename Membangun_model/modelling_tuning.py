import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import optuna
import dagshub
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from optuna.visualization.matplotlib import plot_optimization_history

# ==========================================
# LOAD ENV (AMAN)
# ==========================================
load_dotenv()

DAGSHUB_USERNAME = "Wriyadi5"
DAGSHUB_REPO_NAME = "Eksperimen_SML_Willy-Riyadi"
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

if DAGSHUB_TOKEN is None:
    raise ValueError("DAGSHUB_TOKEN tidak ditemukan di .env")
DAGSHUB_TOKEN = DAGSHUB_TOKEN.strip()
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN


def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
    }

    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return f1_score(y_test, y_pred, average='macro', zero_division=0)


def run_optuna_modelling():
    print("Memulai eksperimen Optuna + MLflow + DagsHub...")

    # MLflow setup
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    mlflow.set_experiment("Stroke_Risk_Optuna")

    # Load data
    data_path = "Membangun_model\stroke_risk_dataset_preprocessing\stroke_risk_dataset_v2_preprocessing.csv"
    df = pd.read_csv(data_path)

    X = df.drop(columns=['at_risk'])
    y = df['at_risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Disable autolog
    mlflow.sklearn.autolog(disable=True)

    # Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=20)

    best_params = study.best_params

    with mlflow.start_run(run_name="RandomForest_Optuna_Manual_Log"):

        # Train best model
        best_model = RandomForestClassifier(**best_params, random_state=42)
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # Logging
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        mlflow.sklearn.log_model(best_model, "rf_optuna_model")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cbar=False)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # Feature Importance
        importances = best_model.feature_importances_
        plt.figure(figsize=(10, 6))
        pd.Series(importances, index=X.columns)\
            .nlargest(10).sort_values().plot(kind='barh')
        plt.title("Top 10 Feature Importances")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()

        # Grafik Riwayat Optimasi Optuna
        # Ini akan menunjukkan bagaimana Optuna mencari parameter terbaik dari waktu ke waktu
        plt.figure(figsize=(10, 6))
        ax = plot_optimization_history(study)
        plt.title("Optuna Optimization History")
        plt.tight_layout()
        plt.savefig("optimization_history.png")
        mlflow.log_artifact("optimization_history.png")
        plt.close()

        print("✅ Optuna tuning selesai & semua log terkirim ke DagsHub!")


if __name__ == "__main__":
    run_optuna_modelling()