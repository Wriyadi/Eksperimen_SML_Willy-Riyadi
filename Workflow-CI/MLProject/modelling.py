import os
import shutil
import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def run_ci_modelling():
    print("Memulai CI Workflow Modelling...")
    
    # 1. Load Data
    data_path = "./stroke_risk_dataset_preprocessing/stroke_risk_dataset_v2_preprocessing.csv"
    df = pd.read_csv(data_path)

    X = df.drop(columns=['at_risk'])
    y = df['at_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Latih Model dengan Best Parameter (Dari eksperimen Kriteria 2)
    model = RandomForestClassifier(
        n_estimators=273, 
        max_depth=20, 
        min_samples_split=5, 
        min_samples_leaf=1, 
        random_state=42
    )
    model.fit(X_train, y_train)

    # 3. Hapus folder model lama jika ada (mencegah error saat overwrite)
    if os.path.exists("saved_model"):
        shutil.rmtree("saved_model")

    # 4. Simpan artefak secara lokal dengan fix versi Starlette
    mlflow.sklearn.save_model(
        model, 
        "saved_model",
        extra_pip_requirements=["starlette<1.0.0"]  # <--- TAMBAHKAN BARIS INI
    )
    print("Artefak model berhasil disimpan di folder 'saved_model'!")

if __name__ == "__main__":
    run_ci_modelling()