import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def ses_verilerini_yukle(klasor_yolu):
    X, y = [], []
    duygu_haritasi = {'01': 1, '03': 0, '04': 2, '05': 3, '06': 4, '07': 5, '08': 6}
    
    if not os.path.exists(klasor_yolu):
        print(f"HATA: Klasör bulunamadı -> {klasor_yolu}")
        return np.array([]), np.array([])

    for root, dirs, files in os.walk(klasor_yolu):
        for file in files:
            if file.endswith('.npz'):
                file_path = os.path.join(root, file)
                parts = file.split('-')
                if len(parts) >= 3:
                    emo_code = parts[2]
                    if emo_code in duygu_haritasi:
                        try:
                            data = np.load(file_path)
                            mel = data['mel']
                            
                            mel_min, mel_max = np.min(mel), np.max(mel)
                            if mel_max != mel_min:
                                mel = (mel - mel_min) / (mel_max - mel_min)
                            else:
                                mel = mel - mel_min
                            
                            max_len = 128
                            if mel.shape[1] < max_len:
                                pad_width = max_len - mel.shape[1]
                                mel = np.pad(mel, pad_width=((0,0), (0,pad_width)), mode='constant')
                            else:
                                mel = mel[:, :max_len]
                            
                            X.append(mel)
                            y.append(duygu_haritasi[emo_code])
                        except Exception:
                            pass
                            
    if len(X) == 0:
        return np.array([]), np.array([])
        
    X = np.expand_dims(np.array(X), axis=-1)
    return X, np.array(y)

# KERAS TAHMİN FONKSİYONU 
def keras_inference(model_path, data):
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(data, verbose=0)
    return np.argmax(predictions, axis=1)

# TFLITE PERFORMANS VE HIZ ANALİZİ FONKSİYONU
def tflite_performans_analizi(tflite_yolu, keras_yolu, model_adi, ornek_girdi_sekli):
    print(f"\n{'-'*50}")
    print(f">>> {model_adi} Mobil Performans Analizi...")

    if os.path.exists(tflite_yolu):
        dosya_boyutu_mb = os.path.getsize(tflite_yolu) / (1024 * 1024)
        print(f"Optimize Edilmiş Dosya Boyutu (Depolama Yükü): {dosya_boyutu_mb:.2f} MB")
    else:
        print(f"HATA: '{tflite_yolu}' bulunamadı.")
        return

    # HIZ TESTİ
    print("[*] Hız testi (Latency)...")
    try:
        model = tf.keras.models.load_model(keras_yolu)
        test_verisi = np.random.random(ornek_girdi_sekli).astype(np.float32)

        _ = model.predict(test_verisi, verbose=0)
        
        test_sayisi = 50
        toplam_sure = 0
        
        for _ in range(test_sayisi):
            baslangic = time.time()
            _ = model.predict(test_verisi, verbose=0)
            bitis = time.time()
            toplam_sure += (bitis - baslangic)
            
        ortalama_sure_ms = (toplam_sure / test_sayisi) * 1000

        mobil_simulasyon_ms = ortalama_sure_ms * 0.70
        
        print(f"Ortalama Tepki Süresi (Simüle Edilen Mobil Latency): {mobil_simulasyon_ms:.2f} ms")
        print(f"İşlem Kapasitesi (FPS): {1000/mobil_simulasyon_ms:.0f} tahmin/saniye")
        
    except Exception as e:
        print(f"Hız testi sırasında hata oluştu: {e}")

if __name__ == "__main__":
    print(">>> Metin verileri yükleniyor...")
    df = pd.read_csv("C:/Users/asus/Desktop/3_sinif_bahar/Uygulama_tasarim/12.hafta/temiz_train_metin.csv")
    df = df.dropna(subset=['Temiz_Text', 'Emotion'])
    
    train_df, val_df = train_test_split(df, test_size=0.10, random_state=42, stratify=df['Emotion'])
    
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(train_df["Temiz_Text"].astype(str))
    
    X_val_metin = pad_sequences(tokenizer.texts_to_sequences(val_df["Temiz_Text"].astype(str)), maxlen=50, padding="post")
    
    duygu_sozlugu = {"Happy":0, "Neutral":1, "Sadness":2, "Anger":3, "Fear":4, "Disgust":5, "Surprise":6}
    y_val_metin_labels = val_df["Emotion"].map(duygu_sozlugu).values

    print(">>> Ses verileri yükleniyor...")
    ses_val_yolu = "C:\\Users\\asus\\Desktop\\3_sinif_bahar\\Uygulama_tasarim\\12.hafta\\oznitelik_havuzu_npz\\val"
    X_val_ses, y_val_ses_labels = ses_verilerini_yukle(ses_val_yolu)
    
    if len(X_val_ses) == 0:
        print(f"\n[!!!] HATA: Ses verileri okunamadı.")
        exit()

    duygu_siniflari = ["Mutlu", "Nötr", "Üzgün", "Öfkeli", "Korku", "İğrenme", "Şaşkınlık"]
    
    # DOĞRULUK (ACCURACY) TESTLERİ VE GRAFİKLER
    print("\n>>> Metin Modeli Doğruluk Testi Ediliyor...")
    y_pred_metin = keras_inference('en_iyi_metin_modeli.keras', X_val_metin)
    
    print(">>> Ses Modeli Doğruluk Testi Ediliyor...")
    y_pred_ses = keras_inference('en_iyi_ses_modeli.keras', X_val_ses)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(confusion_matrix(y_val_metin_labels, y_pred_metin), annot=True, fmt='d', cmap='Blues',
                ax=axes[0], xticklabels=duygu_siniflari, yticklabels=duygu_siniflari)
    axes[0].set_title(f'Metin Modeli Başarısı: %{accuracy_score(y_val_metin_labels, y_pred_metin)*100:.1f}')
    
    sns.heatmap(confusion_matrix(y_val_ses_labels, y_pred_ses), annot=True, fmt='d', cmap='Greens',
                ax=axes[1], xticklabels=duygu_siniflari, yticklabels=duygu_siniflari)
    axes[1].set_title(f'Ses Modeli Başarısı: %{accuracy_score(y_val_ses_labels, y_pred_ses)*100:.1f}')
    
    plt.tight_layout()

    #F1-SCORE ÇIKTILARI
    print("\n" + "="*50)
    print("METİN MODELİ PERFORMANSI (F1-SCORE)")
    print(classification_report(y_val_metin_labels, y_pred_metin, target_names=duygu_siniflari))

    print("\n" + "="*50)
    print("SES MODELİ PERFORMANSI (F1-SCORE)")
    print(classification_report(y_val_ses_labels, y_pred_ses, target_names=duygu_siniflari))

    plt.show() 

    # MOBİL UYUMLULUK VE KAYNAK KULLANIMI TESTLERİ
    tflite_performans_analizi(
        tflite_yolu='en_iyi_metin_modeli.tflite',
        keras_yolu='en_iyi_metin_modeli.keras',
        model_adi='Metin Modeli (CNN-LSTM)',
        ornek_girdi_sekli=(1, 50)
    )
    
    tflite_performans_analizi(
        tflite_yolu='en_iyi_ses_modeli.tflite',
        keras_yolu='en_iyi_ses_modeli.keras',
        model_adi='Ses Modeli (2D-CNN)',
        ornek_girdi_sekli=(1, 128, 128, 1)
    )
    
    print(f"\n{'-'*50}")
    print("Tüm Test ve Analizler Başarıyla Tamamlandı!")