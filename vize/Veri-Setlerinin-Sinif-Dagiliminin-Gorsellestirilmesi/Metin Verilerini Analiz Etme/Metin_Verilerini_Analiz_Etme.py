import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("metinVeriSeti.csv")

# FORMAT KONTROLÜ
print("--- İLK 5 SATIR ---")
print(df.head())
print("\n--- SÜTUN BİLGİLERİ ---")
print(df.info())  # Null (boş) değer var mı

# DENGESİZLİK ANALİZİ
duygu_sayilari = df['Emotion'].value_counts()
print("\n--- DUYGU DAĞILIMI ---")
print(duygu_sayilari)

# Grafiğe dök (görsel)
plt.figure(figsize=(10,6))
sns.barplot(x=duygu_sayilari.index, y=duygu_sayilari.values)
plt.title("Metin Veri Seti Duygu Dağılımı")
plt.show()

# GÜRÜLTÜ KONTROLÜ
# Tekrar eden satır var mı?
print(f"\nTekrar eden satır sayısı: {df.duplicated().sum()}")

# Boş veri var mı?
print(f"Boş (Null) veri sayısı:\n{df.isnull().sum()}")