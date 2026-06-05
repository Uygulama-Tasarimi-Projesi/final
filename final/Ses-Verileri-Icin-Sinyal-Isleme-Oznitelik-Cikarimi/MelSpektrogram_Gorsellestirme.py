import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

npz_yolu ="03-01-01-01-01-01-02_features.npz"

data = np.load(npz_yolu)
mel_db = data['mel']
mfcc = data['mfcc']

# MEL-SPEKTROGRAM (Duygu Haritası)
plt.figure(figsize=(10, 6))
img1 = librosa.display.specshow(mel_db, sr=22050, x_axis='time', y_axis='mel', cmap='magma')
plt.colorbar(img1, format='%+2.0f dB')
plt.title('Mel-Frequency Spectrogram', fontsize=14)
plt.xlabel('Zaman (sn)', fontsize=12)
plt.ylabel('Frekans (Hz)', fontsize=12)
plt.tight_layout()
plt.show()

# MFCC (Tınısal Karakteristikler)
plt.figure(figsize=(10, 6))
img2 = librosa.display.specshow(mfcc, sr=22050, x_axis='time', cmap='viridis')
plt.colorbar(img2)
plt.title('Mel-Frequency Cepstral Coefficients (MFCC)', fontsize=14)
plt.xlabel('Zaman (sn)', fontsize=12)
plt.ylabel('MFCC Katsayıları', fontsize=12)
plt.tight_layout()
plt.show()
