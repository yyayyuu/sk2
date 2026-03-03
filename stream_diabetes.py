# =========================
# 0️⃣ Library
# =========================
import pandas as pd
import pickle
from collections import Counter
from xgboost import XGBClassifier
from imblearn.under_sampling import TomekLinks
import warnings
warnings.filterwarnings('ignore')

# =========================
# 1️⃣ Load dataset training
# =========================
data = pd.read_csv("diabetes_dataset.csv")  # dataset asli dengan target 'diabetes'
X = data.drop("diabetes", axis=1)
y = data["diabetes"]

# =========================
# 2️⃣ One-Hot Encoding kolom kategorikal
# =========================
categorical_cols = ["gender", "smoking_history", "location"]  # semua kolom kategori
for col in categorical_cols:
    X[col] = X[col].astype(str)

X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Simpan daftar kolom training
model_columns = X_encoded.columns.tolist()
pickle.dump(model_columns, open("model_columns.pkl", "wb"))
print("✅ model_columns.pkl berhasil dibuat")

# =========================
# 3️⃣ Balancing dengan TomekLinks
# =========================
tl = TomekLinks()
X_res, y_res = tl.fit_resample(X_encoded, y)
print("Distribusi target setelah TomekLinks:", Counter(y_res))

# =========================
# 4️⃣ Latih XGBoost
# =========================
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb.fit(X_res, y_res)
print("✅ Model XGBoost dengan TomekLinks sudah dilatih")

# Simpan model
pickle.dump(xgb, open("model_xgb_tomek.pkl", "wb"))
print("✅ model_xgb_tomek.pkl berhasil dibuat")

# =========================
# 5️⃣ Prediksi data baru
# =========================
data_baru = pd.read_csv("diabetes_dataset.csv")  # dataset baru tanpa target

# One-Hot Encoding kolom kategorikal
for col in categorical_cols:
    data_baru[col] = data_baru[col].astype(str)
data_baru_encoded = pd.get_dummies(data_baru, columns=categorical_cols)

# Tambahkan kolom yang hilang agar sesuai training
for c in model_columns:
    if c not in data_baru_encoded.columns:
        data_baru_encoded[c] = 0

# Urutkan kolom sesuai training
data_baru_encoded = data_baru_encoded[model_columns]

# Prediksi
model_xgb = pickle.load(open("model_xgb_tomek.pkl", "rb"))
data_baru["prediksi_diabetes"] = model_xgb.predict(data_baru_encoded)

# =========================
# 6️⃣ Simpan hasil prediksi
# =========================
data_baru.to_csv("hasil_prediksi_xgb_tomek.csv", index=False)
print("✅ Hasil prediksi tersimpan di 'hasil_prediksi_xgb_tomek.csv'")