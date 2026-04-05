import requests
import time
import random
import pandas as pd

EXPORTER_URL = "http://localhost:8000/predict"

print("Memuat dataset untuk simulasi inference...")
# Sesuaikan path ini dengan lokasi file CSV preprocessing Anda
data_path = "../Workflow-CI/MLProject/stroke_risk_dataset_preprocessing/stroke_risk_dataset_v2_preprocessing.csv"

try:
    df = pd.read_csv(data_path)
    # Buang kolom target karena kita hanya butuh fitur untuk prediksi
    if 'at_risk' in df.columns:
        X = df.drop(columns=['at_risk'])
    else:
        X = df
        
    # Ubah dataframe menjadi bentuk dictionary (JSON)
    records = X.to_dict(orient='records')
    print(f"Berhasil memuat {len(records)} baris data yang valid!")
except Exception as e:
    print(f"Error memuat data: {e}")
    print("Pastikan path file CSV-nya benar!")
    exit()

print("Memulai simulasi inference traffic...")

while True:
    # Pilih satu baris data aktual secara acak
    random_record = random.choice(records)
    
    payload = {
        "dataframe_records": [random_record]
    }
    
    try:
        # Kirim ke Prometheus Exporter
        response = requests.post(EXPORTER_URL, json=payload)
        print(f"Prediksi: {response.status_code} | Result: {response.text.strip()}")
    except Exception as e:
        print(f"Gagal mengirim request: {e}")
        
    # Jeda acak agar terlihat seperti traffic asli
    time.sleep(random.uniform(1, 3))