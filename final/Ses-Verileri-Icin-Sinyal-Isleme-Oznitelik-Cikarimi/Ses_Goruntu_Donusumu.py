import librosa
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
input_base = os.path.join(current_dir, "islenmis_sesler_npy")
output_base = os.path.join(current_dir, "cnn_hazir_veriler")

def ses_goruntu_donusturucu():
    total_processed = 0
    for root, dirs, files in os.walk(input_base):
        for file_name in files:
            if file_name.endswith(".npy"):
                input_path = os.path.join(root, file_name)
                
                # Hiyerarşiyi Korumak(Mirroring)
                relative_path = os.path.relpath(root, input_base)
                output_folder = os.path.join(output_base, relative_path)
                
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                try:
                    # Temizlenmiş Sinyali Yükleme
                    y = np.load(input_path)
                    sr = 22050

                    # Mel-Spektrogram Oluşturma (Ses-Görüntü Dönüşümü)
                    # n_mels=128, görüntünün 'yüksekliğini' (frekans çözünürlüğünü) belirler.
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    
                    # Enerji Ölçeklendirme (Görsel Netlik için dB Dönüşümü)
                    # Bu işlem piksellerin enerjisini normalize eder.
                    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    save_name = file_name.replace(".npy", "_cnn_input.npy")
                    np.save(os.path.join(output_folder, save_name), mel_db)

                    total_processed += 1
                    if total_processed % 100 == 0:
                        print(f">>> {total_processed} adet görsel matris oluşturuldu.")

                except Exception as e:
                    print(f"Hata: {file_name} işlenirken bir sorun oluştu: {e}")

    print("-" * 30)
    print(f"Toplam {total_processed} dosyanın Ses-Görüntü Dönüşümü tamamlandı.")

if __name__ == "__main__":
    ses_goruntu_donusturucu()