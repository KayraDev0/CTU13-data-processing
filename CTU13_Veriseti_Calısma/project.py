import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pyshark

# Veriyi yükle
attack_df = pd.read_csv("CTU13_Attack_Traffic.csv")
normal_df = pd.read_csv("CTU13_Normal_Traffic.csv")


# Eksik verileri medyan ile doldur
attack_df.fillna(attack_df.median(), inplace=True)
normal_df.fillna(normal_df.median(), inplace=True)

# Etiketler ekle
attack_df["Label"] = 1  # Saldırı trafiği
normal_df["Label"] = 0  # Normal trafik

# Verileri birleştir
full_df = pd.concat([attack_df, normal_df], ignore_index=True)

# Gereksiz sütunları çıkar
full_df = full_df.drop(columns=["Src IP", "Dst IP", "Timestamp"], errors='ignore')

# Korelasyon analizi
corr_matrix = full_df.corr()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
print("Çıkarılması düşünülebilecek yüksek korelasyonlu sütunlar:", high_corr_features)

# Yüksek korelasyonlu sütunları çıkar
drop_columns = [
    'Fwd Pkt Len Std', 'Bwd Pkt Len Std', 'Fwd IAT Tot', 'Bwd Header Len',
    'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Max', 'Pkt Len Std', 'Pkt Size Avg',
    'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Act Data Pkts', 'Active Min',
    'Idle Max', 'Idle Min'
]
full_df = full_df.drop(columns=drop_columns, errors='ignore')

# Veriyi ölçeklendir
scaler = StandardScaler()
numeric_cols = full_df.select_dtypes(include=['int64', 'float64']).columns
full_df[numeric_cols] = scaler.fit_transform(full_df[numeric_cols])

# Özellikler ve etiketleri ayır
X = full_df.drop(columns=["Label"])  # Özellikler
y = full_df["Label"]  # Etiket


# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = y_train.astype(int) #float türünde olduğundan integer olarak değiştirdim
y_test = y_test.astype(int)

# Veriyi ölçeklendir
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Modeli
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)

# Modeli değerlendirme
y_pred = model.predict(X_test_scaled)

# Doğruluk oranını hesapla
accuracy = accuracy_score(y_test, y_pred)
print("Model Doğruluk Oranı:", accuracy)

# Karışıklık matrisi
cm = confusion_matrix(y_test, y_pred)
print("Karışıklık Matrisi:")
print(cm)


