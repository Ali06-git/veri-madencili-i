# veri-madencili-i
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

# 1. Veri Setini Yükleme
print("Veri seti yükleniyor...")
# OpenML üzerinden UCI Steel Plates Faults veri setini çekiyoruz
steel_data = fetch_openml(name='steel-plates-fault', version=1, as_frame=True, parser='auto')
df = steel_data.frame

# 2. Hedef ve Bağımsız Değişkenlerin Ayrılması
target_col = steel_data.target_names[0]
bagimsiz_degiskenler = [col for col in df.columns if col != target_col]

# 3. Tanımlayıcı İstatistikler
print("\n--- Tanımlayıcı İstatistikler ---")
# Temel metriklerin yanına çarpıklık (skewness) ve basıklık (kurtosis) ekliyoruz
desc_stats = df[bagimsiz_degiskenler].describe().T
desc_stats['skewness'] = df[bagimsiz_degiskenler].skew(numeric_only=True)
desc_stats['kurtosis'] = df[bagimsiz_degiskenler].kurtosis(numeric_only=True)
print(desc_stats.round(2))

# 4. Sınıf Dengesizliği (Class Imbalance) Kontrolü
print("\n--- Kusur Sınıflarının Dağılımı ---")
sinif_dagilimi = df[target_col].value_counts()
print(sinif_dagilimi)

# 5. Korelasyon Analizi
print("\n--- Korelasyon Analizi ---")
corr_matrix = df[bagimsiz_degiskenler].corr(numeric_only=True)

# Birbiriyle yüksek korelasyonlu (|r| > 0.8) bağımsız değişkenleri bulma
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.8:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

print("\nYüksek Korelasyona Sahip Değişken Çiftleri (|r| > 0.8):")
for pair in high_corr_pairs:
    print(f"{pair[0]} - {pair[1]}: {pair[2]:.2f}")

# 6. Korelasyon Matrisi Görselleştirme (Heatmap)
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Çelik Levha Özellikleri - Korelasyon Matrisi', fontsize=16)
plt.tight_layout()
plt.show()

