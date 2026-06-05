import librosa
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "RAVDESS_Dataset")
output_base = os.path.join(current_dir, "islenmis_sesler_npy")

def hiyerarsik_ses_isle():
    processed_count = 0
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith(".wav"):
                input_file_path = os.path.join(root, file_name)
                
                relative_path = os.path.relpath(root, dataset_path)
                output_folder = os.path.join(output_base, relative_path)
                
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                
                try:
                    #  Dijital Sinyale Dönüştürme
                    y, sr = librosa.load(input_file_path, sr=22050)
                    
                    #  Gürültü Temizleme ve Ön İşleme
                    y_trimmed, _ = librosa.effects.trim(y)
                    y_norm = librosa.util.normalize(y_trimmed)
                    
                    # Binary format (.npy) olarak kaydet
                    save_path = os.path.join(output_folder, file_name.replace(".wav", ".npy"))
                    np.save(save_path, y_norm)
                    
                    processed_count += 1
                except Exception as e:
                    print(f"Hata: {file_name} -> {e}")

    print(f"Toplam {processed_count} dosya kaydedildi.")

if __name__ == "__main__":
    hiyerarsik_ses_isle()