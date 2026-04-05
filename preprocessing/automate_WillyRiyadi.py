import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def run_preprocessing(input_filepath, output_filepath):
    """
    Fungsi untuk melakukan preprocessing secara otomatis dan 
    mengembalikan dataframe yang siap dilatih.
    """
    print(f"Memulai preprocessing untuk file: {input_filepath}")
    
    # 1. Load Data
    try:
        data = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(f"Error: File {input_filepath} tidak ditemukan. Pastikan file berada di folder yang sama.")
        return None

    # 2. Definisikan Target dan Fitur yang akan di-drop
    target_col = 'at_risk'
    cols_to_drop = [target_col, 'stroke_risk_percentage']
    existing_cols_to_drop = [col for col in cols_to_drop if col in data.columns]

    # Pisahkan fitur (X) dan target (y)
    X = data.drop(columns=existing_cols_to_drop)
    y = data[target_col]

    # 3. Penyesuaian tipe data (Sesuai eksperimen sebelumnya)
    int_cols = X.select_dtypes(include=['int64']).columns
    X[int_cols] = X[int_cols].astype('float64')

    # 4. Deteksi kolom kategorikal dan numerik
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['float64']).columns.tolist()

    # 5. Bangun Pipeline Preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # WAJIB: sparse_output=False agar hasil bisa dikonversi kembali ke DataFrame Pandas
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 6. Fit & Transform Data (Terapkan preprocessing)
    print("Melakukan transformasi data (Imputasi, Scaling, Encoding)...")
    X_preprocessed = preprocessor.fit_transform(X)

    # 7. Kembalikan ke format DataFrame
    # Ambil nama kolom baru setelah di-OneHotEncode
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + cat_feature_names.tolist()

    df_preprocessed = pd.DataFrame(X_preprocessed, columns=all_feature_names)

    # 8. Gabungkan kembali dengan kolom target agar siap dilatih
    df_preprocessed[target_col] = y.values

    # 9. Simpan ke CSV baru
    df_preprocessed.to_csv(output_filepath, index=False)
    print(f"Preprocessing selesai! Data bersih berhasil disimpan di: {output_filepath}")
    
    return df_preprocessed

# Blok ini akan tereksekusi jika file dijalankan langsung dari terminal
if __name__ == "__main__":
    # Path dataset RAW (sesuai struktur repo)
    input_filepath = "../stroke_risk_dataset_raw/stroke_risk_dataset_v2_raw.csv"
    
    # Path output preprocessing
    output_filepath = "./stroke_risk_dataset_preprocessing/stroke_risk_dataset_v2_preprocessing.csv"
    
    run_preprocessing(input_filepath, output_filepath)