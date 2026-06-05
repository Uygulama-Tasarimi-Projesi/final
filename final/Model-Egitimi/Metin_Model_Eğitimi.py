import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Concatenate, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

print("--- METİN MODELİ (CNN-LSTM) EĞİTİM SÜRECİ ---")

# MODEL MİMARİSİ OLUŞTURMA FONKSİYONU
def cnn_lstm_modeli_olustur():
    VOCAB_SIZE = 10000    # Sözlük boyutu
    MAX_LENGTH = 50       # Cümle uzunluğu
    EMBEDDING_DIM = 128   # Kelime vektör boyutu
    NUM_CLASSES = 7       # 7 temel duygu sınıfı 

    # GİRİŞ VE GÖMME (EMBEDDING) KATMANLARI
    input_layer = Input(shape=(MAX_LENGTH,), name="Metin_Giris_Katmani")
    embedding_layer = Embedding(input_dim=VOCAB_SIZE, 
                                output_dim=EMBEDDING_DIM, 
                                name="Kelime_Gomme_Kati")(input_layer)

    # ÇOKLU N-GRAM CNN MİMARİSİ 
    conv_2 = Conv1D(filters=64, kernel_size=2, activation='relu', padding='same', name="Conv1D_Bigram")(embedding_layer)
    pool_2 = MaxPooling1D(pool_size=2, name="MaxPool_Bigram")(conv_2)

    conv_3 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name="Conv1D_Trigram")(embedding_layer)
    pool_3 = MaxPooling1D(pool_size=2, name="MaxPool_Trigram")(conv_3)

    conv_4 = Conv1D(filters=64, kernel_size=4, activation='relu', padding='same', name="Conv1D_4gram")(embedding_layer)
    pool_4 = MaxPooling1D(pool_size=2, name="MaxPool_4gram")(conv_4)

    # Özniteliklerin Birleştirilmesi
    cnn_features = Concatenate(axis=-1, name="N_Gram_Birlestirme")([pool_2, pool_3, pool_4])

    # LSTM KATMANLARI
    lstm_1 = LSTM(units=128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, name="LSTM_Katmani_1")(cnn_features)
    lstm_2 = LSTM(units=64, dropout=0.3, recurrent_dropout=0.3, name="LSTM_Katmani_2")(lstm_1)

    # ÇIKIŞ KATMANI
    dropout_layer = Dropout(rate=0.4, name="Dropout_Genel")(lstm_2)
    output_layer = Dense(units=NUM_CLASSES, activation='softmax', name="Duygu_Cikisi_7_Sinif")(dropout_layer)

    final_model = Model(inputs=input_layer, outputs=output_layer, name="CNN_LSTM_Hibrit_Duygu_Modeli")
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return final_model

# GRAFİK ÇİZDİRME FONKSİYONU
def metin_egitim_grafiklerini_ciz(history, model_adi="CNN-LSTM_Metin_Modeli"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Doğruluk (Accuracy) Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Eğitim Başarısı (Train Acc)')
    plt.plot(epochs, val_acc, 'r', label='Doğrulama Başarısı (Val Acc)')
    plt.title(f'{model_adi} - Başarı Oranı')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Kayıp (Loss) Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Eğitim Kaybı (Train Loss)')
    plt.plot(epochs, val_loss, 'r', label='Doğrulama Kaybı (Val Loss)')
    plt.title(f'{model_adi} - Kayıp Oranı')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

if __name__ == "__main__":
    print("\n>>> Veriler Yükleniyor ve Yeniden Bölümleniyor...")

    df = pd.read_csv("C:/Users/asus/Desktop/3_sinif_bahar/Uygulama_tasarim/11.hafta/temiz_train_metin.csv")
    df = df.dropna(subset=['Temiz_Text', 'Emotion'])

    train_df, val_df = train_test_split(df, test_size=0.10, random_state=42, stratify=df['Emotion'])

    print(f"> Eğitim için kullanılacak veri: {len(train_df)} adet")
    print(f"> Doğrulama (Val) için ayrılan veri: {len(val_df)} adet")

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(train_df["Temiz_Text"].astype(str))
    
    X_train = pad_sequences(tokenizer.texts_to_sequences(train_df["Temiz_Text"].astype(str)), maxlen=50, padding="post")
    X_val = pad_sequences(tokenizer.texts_to_sequences(val_df["Temiz_Text"].astype(str)), maxlen=50, padding="post")

    # One-Hot Encoding
    duygu_sozlugu = {"Happy":0, "Neutral":1, "Sadness":2, "Anger":3, "Fear":4, "Disgust":5, "Surprise":6}
    
    y_train_num = train_df["Emotion"].map(duygu_sozlugu)
    y_val_num = val_df["Emotion"].map(duygu_sozlugu)
    
    y_train = tf.keras.utils.to_categorical(y_train_num, num_classes=7)
    y_val = tf.keras.utils.to_categorical(y_val_num, num_classes=7)

    print("\n>>> Model Oluşturuluyor...")
    text_model = cnn_lstm_modeli_olustur()
    
    print("\n>>> Çalıştırılıyor...")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint_text = ModelCheckpoint('en_iyi_metin_modeli.keras', monitor='val_accuracy', save_best_only=True, mode='max')

    text_history = text_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop, checkpoint_text]
    )
    
    print(">>> Metin eğitim modülü tamamlandı. Grafikler Çiziliyor...")
    metin_egitim_grafiklerini_ciz(text_history)