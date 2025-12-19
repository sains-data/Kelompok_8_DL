import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv  # Mengaktifkan fitur eksperimen
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib  # Untuk menyimpan model

# 1) Load Data
df = pd.read_excel("Data_Response_Gform_Deep_Learning (1).xlsx")

# Preprocessing (Clean and prepare data)
df.columns = df.columns.str.strip()
df = df.rename(columns={ 
    "Berapa usia Anda saat ini?": "usia", 
    "Pilih jenis kelamin Anda": "jenis_kelamin", 
    "Apakah Anda memiliki penyakit fisik (misalnya penyakit kronis)?": "penyakit_fisik",
    "Berapa lama rata-rata waktu penggunaan layar Anda setiap hari (HP, laptop, TV, dll)?": "screen_time",
    "Seberapa sering Anda berolahraga?": "frekuensi_olahraga",
    "Apakah Anda memiliki kebiasaan merokok atau minum alkohol?": "rokok_alkohol",
    "sleep_time": "sleep_time" 
})

# Mengubah kolom usia untuk hanya mengambil angka
df["usia"] = df["usia"].astype(str).str.extract(r'(\d+)').astype(float)

# Mengonversi kolom 'screen_time' ke nilai numerik yang sesuai
# Fungsi untuk mengonversi input waktu layar berdasarkan pedoman kesehatan
def convert_screen_time(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if "< 2" in x:
        return 1.5  # Menetapkan 2 jam sebagai batas yang disarankan
    elif "2-4" in x:
        return 3.0  # Rata-rata waktu untuk kategori 2-4 jam
    elif "> 4" in x:
        return 5.0  # Rata-rata waktu untuk kategori lebih dari 4 jam
    else:
        return np.nan  # Menghindari nilai yang tidak sesuai

df["screen_time"] = df["screen_time"].apply(convert_screen_time)

features = ["usia", "jenis_kelamin", "penyakit_fisik", "screen_time", "frekuensi_olahraga", "rokok_alkohol"]
X = df[features]
y = df["sleep_time"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Pipeline
numeric_features = ["usia", "screen_time"]
categorical_features = [c for c in X.columns if c not in numeric_features]

numeric_transformer = Pipeline([ 
    ("imputer", SimpleImputer(strategy="median")), 
    ("scaler", RobustScaler()) 
])

categorical_transformer = Pipeline([ 
    ("imputer", SimpleImputer(strategy="most_frequent")), 
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)) 
])

preprocessor = ColumnTransformer([ 
    ("num", numeric_transformer, numeric_features), 
    ("cat", categorical_transformer, categorical_features) 
])

mlp = MLPRegressor(
    max_iter=3000,
    early_stopping=True,
    random_state=42,
    validation_fraction=0.2  # Menambahkan validation_fraction untuk menghindari kesalahan validasi kecil
)

pipeline = Pipeline([ 
    ("preprocessor", preprocessor), 
    ("model", TransformedTargetRegressor(regressor=mlp, transformer=RobustScaler())) 
])

# Hyperparameter Tuning menggunakan HalvingGridSearchCV
param_grid = {
    "model__regressor__hidden_layer_sizes": [(32,), (64,), (64,32)],
    "model__regressor__activation": ["relu", "tanh"],
    "model__regressor__alpha": [0.0001, 0.001],
    "model__regressor__learning_rate_init": [0.001, 0.0005],
    "model__regressor__solver": ["adam", "sgd"],
    "model__regressor__max_iter": [1, 5]
}

search = HalvingGridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    factor=2,
    verbose=2,
    error_score='raise'  # Menambahkan error_score untuk memudahkan debugging
)

# Melatih model dengan data
search.fit(X_train, y_train)

# Simpan model ke file
joblib.dump(search.best_estimator_, 'model.pkl')  # Menyimpan model terbaik ke file
print("Model telah disimpan ke 'model.pkl'")
