import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

npy_yolu = "C:\\Users\\asus\\Desktop\\3_sinif_bahar\\Uygulama_tasarim\\8.hafta\\ses goruntu donuşumu\\cnn_hazir_veriler\\train\\Actor_02\\03-01-01-01-01-01-02_cnn_input.npy"
if os.path.exists(npy_yolu):
    ses_sinyali = np.load(npy_yolu)
    
    # Teknik Bilgiler
    print(f"--- Dosya Analizi: {os.path.basename(npy_yolu)} ---")
    print(f"Sinyal Uzunluğu (Örnek Sayısı): {len(ses_sinyali)}")
    print(f"Veri Tipi: {ses_sinyali.dtype}")
    print(f"Maksimum Genlik: {np.max(ses_sinyali)}") # Normalizasyon kontrolü
    print(f"Minimum Genlik: {np.min(ses_sinyali)}")
    
    # Görselleştirme (Duygu Karakteristiği Analizi)
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(ses_sinyali, sr=22050, color='blue')
    
    plt.title(f"İşlenmiş Dijital Sinyal")
    plt.xlabel("Zaman (saniye)")
    plt.ylabel("Normalize Edilmiş Genlik")
    plt.grid(True, alpha=0.3)
    plt.show()

else:
    print("Dosya yolu bulunamadı")