import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

duygu_dosyalari = {
    "Nötr": "C:\\Users\\asus\\Desktop\\3_sinif_bahar\\Uygulama_tasarim\\8.hafta\\akustik oznitelik çıkarımı\\oznitelik_havuzu_npz\\train\\Actor_02\\03-01-01-01-01-01-02_features.npz",
    "Mutlu": "C:\\Users\\asus\\Desktop\\3_sinif_bahar\\Uygulama_tasarim\\8.hafta\\akustik oznitelik çıkarımı\\oznitelik_havuzu_npz\\train\\Actor_02\\03-01-03-01-01-01-02_features.npz",
    "Üzgün": "C:\\Users\\asus\\Desktop\\3_sinif_bahar\\Uygulama_tasarim\\8.hafta\\akustik oznitelik çıkarımı\\oznitelik_havuzu_npz\\train\\Actor_02\\03-01-04-01-01-01-02_features.npz",
    "Sinirli": "C:\\Users\\asus\\Desktop\\3_sinif_bahar\\Uygulama_tasarim\\8.hafta\\akustik oznitelik çıkarımı\\oznitelik_havuzu_npz\\train\\Actor_02\\03-01-05-01-01-01-02_features.npz",
    "Korkmuş": "C:\\Users\\asus\\Desktop\\3_sinif_bahar\\Uygulama_tasarim\\8.hafta\\akustik oznitelik çıkarımı\\oznitelik_havuzu_npz\\train\\Actor_02\\03-01-06-01-01-01-02_features.npz",
    "İğrenmiş": "C:\\Users\\asus\\Desktop\\3_sinif_bahar\\Uygulama_tasarim\\8.hafta\\akustik oznitelik çıkarımı\\oznitelik_havuzu_npz\\train\\Actor_02\\03-01-07-01-01-01-02_features.npz",
    "Şaşkın": "C:\\Users\\asus\\Desktop\\3_sinif_bahar\\Uygulama_tasarim\\8.hafta\\akustik oznitelik çıkarımı\\oznitelik_havuzu_npz\\train\\Actor_02\\03-01-08-01-01-01-02_features.npz"
}

for duygu_adi, npz_yolu in duygu_dosyalari.items():
    data = np.load(npz_yolu)
    mfcc = data['mfcc']
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))

    fig.suptitle(f'{duygu_adi}', fontsize=28, fontweight='bold', color='darkblue')

    # MFCC (Tınısal Karakteristikler)
    img2 = librosa.display.specshow(mfcc, sr=22050, x_axis='time', cmap='viridis', ax=axes)
    fig.colorbar(img2, ax=axes)
    axes.set_xlabel('Zaman (sn)', fontsize=12)
    axes.set_ylabel('MFCC Katsayıları', fontsize=12)

    plt.tight_layout()
    plt.show()