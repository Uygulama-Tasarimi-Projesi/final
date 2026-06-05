import pandas as pd
from sklearn.model_selection import train_test_split

# VERİYİ YÜKLE
df = pd.read_csv("metinVeriSeti.csv")

# STRATIFIED SPLIT - Sınıf dengesini koruyarak
# %80, %20 ayırma
# Kümelerine aynı oranda dağılmasını sağlar.
train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['Emotion'])

# Kalan %20'lik kısmı tam ortadan ikiye bölerek %10 Validation ve %10 Test
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['Emotion'])

print(f"Toplam Veri: {len(df)}")
print(f"Eğitim (Train) Seti: {len(train_df)} satır")
print(f"Doğrulama (Validation) Seti: {len(val_df)} satır")
print(f"Test Seti: {len(test_df)} satır")

print("\n--- Train Seti Duygu Dağılımı ---")
print(train_df['Emotion'].value_counts())

# İndeksleri kaydetmemek için index=False yapıyoruz
train_df.to_csv("train_metin.csv", index=False)
val_df.to_csv("val_metin.csv", index=False)
test_df.to_csv("test_metin.csv", index=False)


print("\nVeriler başarıyla train_metin.csv, val_metin.csv ve test_metin.csv olarak kaydedildi!")
