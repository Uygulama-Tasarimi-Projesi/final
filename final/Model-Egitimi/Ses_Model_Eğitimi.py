import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

print("SES MODELİ (2D-CNN) EĞİTİM SÜRECİ")

# SES MODELİ MİMARİSİ
def ses_analizi_cnn_modeli_olustur():
    input_shape = (128, 128, 1)
    model = models.Sequential(name="Ses_Analizi_2D_CNN_Modeli")

    # Öznitelik Çıkarımı Katmanları
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name="Convolution_1")) 
    model.add(layers.MaxPooling2D((2, 2), name="Pooling_1"))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', name="Convolution_2"))
    model.add(layers.MaxPooling2D((2, 2), name="Pooling_2"))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', name="Convolution_3"))
    
    # Stabilizasyon ve Final Katmanları
    model.add(layers.Dropout(0.3, name="Dropout_1")) 
    model.add(layers.Flatten(name="Flatten_Layer")) 
    model.add(layers.Dense(64, activation='relu', name="Dense_Hidden")) 
    model.add(layers.Dropout(0.3, name="Dropout_Final")) 
    
    # Çıkış Katmanı
    model.add(layers.Dense(7, activation='softmax', name="Duygu_Cikisi_7_Sinif"))

    # Learning Rate Optimizasyonu
    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def ses_egitim_grafiklerini_ciz(history, model_adi="2D-CNN_Ses_Modeli"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Eğitim Başarısı (Train Acc)')
    plt.plot(epochs, val_acc, 'r', label='Doğrulama Başarısı (Val Acc)')
    plt.title(f'{model_adi} - Başarı Oranı')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Eğitim Kaybı (Train Loss)')
    plt.plot(epochs, val_loss, 'r', label='Doğrulama Kaybı (Val Loss)')
    plt.title(f'{model_adi} - Kayıp Oranı')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

# NPZ VERİ YÜKLEYİCİ VE MİN-MAX NORMALİZASYONU
def ses_verilerini_yukle(klasor_yolu):
    X = []
    y = []
    
    duygu_haritasi = {'01': 1, '03': 0, '04': 2, '05': 3, '06': 4, '07': 5, '08': 6}
    
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
                            
                            # Min-Max Scaling (Normalizasyon)
                            mel_min = np.min(mel)
                            mel_max = np.max(mel)
                            if mel_max != mel_min:
                                mel = (mel - mel_min) / (mel_max - mel_min)
                            else:
                                mel = mel - mel_min 
                            
                            # Padding / Cropping işlemleri (128x128 sabitleme)
                            max_len = 128
                            if mel.shape[1] < max_len:
                                pad_width = max_len - mel.shape[1]
                                mel = np.pad(mel, pad_width=((0,0), (0,pad_width)), mode='constant')
                            else:
                                mel = mel[:, :max_len]
                            
                            X.append(mel)
                            y.append(duygu_haritasi[emo_code])
                        except Exception as e:
                            print(f"Hata ({file}): {e}")
                            
    X = np.expand_dims(np.array(X), axis=-1)
    y = tf.keras.utils.to_categorical(np.array(y), num_classes=7)
    
    return X, y

if __name__ == "__main__":
    base_dir = "C:\\Users\\asus\\Desktop\\3_sinif_bahar\\Uygulama_tasarim\\11.hafta\\oznitelik_havuzu_npz"
    
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    if not os.path.exists(train_dir):
        print(f"\n[!!!] HATA: Klasör bulunamadı!\nLütfen şu yolun doğruluğunu kontrol et:\n{train_dir}")
        exit()
    
    print("\n>>> Eğitim Verileri (Train) Yükleniyor ve Normalize Ediliyor...")
    X_train, y_train = ses_verilerini_yukle(train_dir)
    print(f"Train Boyutu: {X_train.shape}, Etiketler: {y_train.shape}")
    
    print("\n>>> Doğrulama Verileri (Val) Yükleniyor ve Normalize Ediliyor...")
    X_val, y_val = ses_verilerini_yukle(val_dir)
    print(f"Val Boyutu: {X_val.shape}, Etiketler: {y_val.shape}")

    print("\n>>> Model Oluşturuluyor...")
    audio_model = ses_analizi_cnn_modeli_olustur()
    
    print("\n>>> Çalıştırılıyor...")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint_audio = ModelCheckpoint('en_iyi_ses_modeli.keras', monitor='val_accuracy', save_best_only=True, mode='max')

    audio_history = audio_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=[early_stop, checkpoint_audio]
    )
    
    print(">>> Ses eğitim modülü tamamlandı. Grafikler Çiziliyor...")
    ses_egitim_grafiklerini_ciz(audio_history)