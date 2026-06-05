import librosa
import numpy as np
import os

# Yolları belirleme
current_dir = os.path.dirname(os.path.abspath(__file__))
input_base = os.path.join(current_dir, "islenmis_sesler_npy")
output_base = os.path.join(current_dir, "oznitelik_havuzu_npy")

def derinlemesine_oznitelik_cikarma():
    total_processed = 0
    for root, dirs, files in os.walk(input_base):
        for file_name in files:
            if file_name.endswith(".npy"):
                input_file_path = os.path.join(root, file_name)
                
                # Çıktı klasör yolunu mirror yaparak oluştur
                relative_path = os.path.relpath(root, input_base)
                output_folder = os.path.join(output_base, relative_path)
                
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                try:
                    # Sinyali yükle
                    y = np.load(input_file_path)
                    sr = 22050

                    # Mel-Spektrogram Hesaplama (Ses-Görüntü Dönüşümü)
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

                    # MFCC Hesaplama (Akustik Karakteristikler)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

                    # .npz formatında sıkıştırarak kaydet
                    save_name = file_name.replace(".npy", "_features.npz")
                    np.savez_compressed(os.path.join(output_folder, save_name), mel=mel_db, mfcc=mfcc)

                    total_processed += 1
                    if total_processed % 100 == 0:
                        print(f">>> {total_processed} dosya başarıyla işlendi.")

                except Exception as e:
                    print(f"Hata: {file_name} -> {e}")

    print("-" * 30)
    print(f"Toplam {total_processed} dosya öznitelik havuzuna eklendi.")
    print(f"Yeni veri havuzu: {output_base}")

if __name__ == "__main__":
    derinlemesine_oznitelik_cikarma()