import tensorflow as tf
from tensorflow.keras import layers, models

def ses_analizi_cnn_modeli_olustur():
    # Mel-Spektrogram giriş boyutu (128x128x1) 
    input_shape = (128, 128, 1)
    model = models.Sequential(name="Ses_Analizi_2D_CNN_Modeli")

    # Öznitelik Çıkarımı Katmanları
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name="Convolution_1")) 
    model.add(layers.MaxPooling2D((2, 2), name="Pooling_1"))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', name="Convolution_2"))
    model.add(layers.MaxPooling2D((2, 2), name="Pooling_2"))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', name="Convolution_3"))
    
    # Stabilizasyon ve Final Katmanları
    
    # Dropout Entegrasyonu Overfitting'i önlemek için stratejik konumlandırma
    model.add(layers.Dropout(0.3, name="Dropout_1")) 

    # Boyut Uyuşmazlığı Hatalarının Çözümü
    # 2D veriyi 1D vektöre düzleştirerek tam bağlantılı katmanlara geçişi sağlar 
    model.add(layers.Flatten(name="Flatten_Layer")) 

    # Ara katman (Dense) ve genelleme yeteneği için tekrar Dropout
    model.add(layers.Dense(64, activation='relu', name="Dense_Hidden")) 
    model.add(layers.Dropout(0.3, name="Dropout_Final")) 

    # Çıkış Katmanı ve Softmax Yapılandırması
    # 7 temel duygu sınıfı 
    model.add(layers.Dense(7, activation='softmax', name="Duygu_Cikisi_7_Sinif"))

    # Modelin Derlenmesi
    # Kayıp Fonksiyonu çok sınıflı yapı için Categorical Crossentropy
    # Optimizasyon için Adam algoritması
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model

if __name__ == "__main__":
    model = ses_analizi_cnn_modeli_olustur()