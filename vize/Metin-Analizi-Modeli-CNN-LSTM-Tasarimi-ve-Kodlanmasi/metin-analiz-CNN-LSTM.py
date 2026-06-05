import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Concatenate, LSTM, Dense, Dropout
from tensorflow.keras.models import Model

# --- HİPERPARAMETRELER ---
VOCAB_SIZE = 10000    # Sözlük boyutu
MAX_LENGTH = 50       # Cümle uzunluğu
EMBEDDING_DIM = 128   # Kelime vektör boyutu
NUM_CLASSES = 7       # 7 temel duygu sınıfı 
#(Mutlu, Nötr, Üzgün, Öfkeli, İğrenme, Korku, Şaşkınlık)

# GİRİŞ VE GÖMME (EMBEDDING) KATMANLARI
input_layer = Input(shape=(MAX_LENGTH,), name="Metin_Giris_Katmani")
embedding_layer = Embedding(input_dim=VOCAB_SIZE, 
                            output_dim=EMBEDDING_DIM, 
                            name="Kelime_Gomme_Kati")(input_layer)

# ÇOKLU N-GRAM CNN MİMARİSİ 
# Bigram
conv_2 = Conv1D(filters=64, kernel_size=2, activation='relu', padding='same', name="Conv1D_Bigram")(embedding_layer)
pool_2 = MaxPooling1D(pool_size=2, name="MaxPool_Bigram")(conv_2)

# Trigram 
conv_3 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name="Conv1D_Trigram")(embedding_layer)
pool_3 = MaxPooling1D(pool_size=2, name="MaxPool_Trigram")(conv_3)

# 4-gram
conv_4 = Conv1D(filters=64, kernel_size=4, activation='relu', padding='same', name="Conv1D_4gram")(embedding_layer)
pool_4 = MaxPooling1D(pool_size=2, name="MaxPool_4gram")(conv_4)

# Özniteliklerin Birleştirilmesi
cnn_features = Concatenate(axis=-1, name="N_Gram_Birlestirme")([pool_2, pool_3, pool_4])

# LSTM KATMANLARI
# İlk LSTM katmanı, diziyi bir sonraki katmana aktarır 
lstm_1 = LSTM(units=128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, name="LSTM_Katmani_1")(cnn_features)

# İkinci LSTM katmanı bağlamı özetler
lstm_2 = LSTM(units=64, dropout=0.3, recurrent_dropout=0.3, name="LSTM_Katmani_2")(lstm_1)

# ÇIKIŞ KATMANI (SINIFLANDIRMA)
dropout_layer = Dropout(rate=0.4, name="Dropout_Genel")(lstm_2)
# 7 duygu sınıfı için softmax aktivasyonu [cite: 49, 76]
output_layer = Dense(units=NUM_CLASSES, activation='softmax', name="Duygu_Cikisi_7_Sinif")(dropout_layer)

final_model = Model(inputs=input_layer, outputs=output_layer, name="CNN_LSTM_Hibrit_Duygu_Modeli")

final_model.compile(optimizer='adam', 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
final_model.summary()
