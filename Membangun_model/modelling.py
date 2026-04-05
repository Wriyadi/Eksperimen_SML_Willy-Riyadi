import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def run_basic_modelling():
    print("Memulai eksperimen Basic Modelling (Lokal & Autolog)...")
    
    # 1. Setup MLflow Tracking Server Lokal
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Stroke_Risk_Basic")

    # 2. Load Data Preprocessing
    data_path = "./stroke_risk_dataset_preprocessing/stroke_risk_dataset_v2_preprocessing.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File {data_path} tidak ditemukan.")
        return

    # 3. Pisahkan Fitur dan Target
    X = df.drop(columns=['at_risk'])
    y = df['at_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Aktifkan MLflow Autolog (Wajib untuk Kriteria Basic)
    mlflow.sklearn.autolog()

    # 5. Latih Model
    with mlflow.start_run(run_name="RandomForest_Basic_Run"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        print("Model Basic berhasil dilatih! Cek http://127.0.0.1:5000 untuk melihat Dashboard.")

if __name__ == "__main__":
    run_basic_modelling()