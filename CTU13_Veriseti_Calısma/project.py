import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pyshark


attack_df = pd.read_csv("CTU13_Attack_Traffic.csv")
normal_df = pd.read_csv("CTU13_Normal_Traffic.csv")



attack_df.fillna(attack_df.median(), inplace=True) #Eksik verileri medyan ile doldur
normal_df.fillna(normal_df.median(), inplace=True)


attack_df["Label"] = 1  #Saldırı trafiği
normal_df["Label"] = 0  #Normal trafik


full_df = pd.concat([attack_df, normal_df], ignore_index=True) #Verileri birleştir


full_df = full_df.drop(columns=["Src IP", "Dst IP", "Timestamp"], errors='ignore') # Gereksiz sütunları çıkar

#Korelasyon analizi
corr_matrix = full_df.corr()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
print("Çıkarılması düşünülebilecek yüksek korelasyonlu sütunlar:", high_corr_features)

#Yüksek korelasyonlu sütunları çıkar
drop_columns = [
    'Fwd Pkt Len Std', 'Bwd Pkt Len Std', 'Fwd IAT Tot', 'Bwd Header Len',
    'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Max', 'Pkt Len Std', 'Pkt Size Avg',
    'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Act Data Pkts', 'Active Min',
    'Idle Max', 'Idle Min'
]
full_df = full_df.drop(columns=drop_columns, errors='ignore')

#Veriyi ölçeklendir
scaler = StandardScaler()
numeric_cols = full_df.select_dtypes(include=['int64', 'float64']).columns
full_df[numeric_cols] = scaler.fit_transform(full_df[numeric_cols])

#Özellikler ve etiketleri ayır
X = full_df.drop(columns=["Label"])
y = full_df["Label"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Veriyi eğit

y_train = y_train.astype(int) #Hata çıktığından float veri tipleri intager olarak aldım
y_test = y_test.astype(int)


X_train_scaled = scaler.fit_transform(X_train) #Veriyi ölçeklendir
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=10000) #Logistic regression modeli
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled) #Modeli değerlendirme


accuracy = accuracy_score(y_test, y_pred) #Doğruluk oranını
print("Model Doğruluk Oranı:", accuracy)


cm = confusion_matrix(y_test, y_pred) #Karışıklık matrisi
print("Karışıklık Matrisi:")
print(cm)


