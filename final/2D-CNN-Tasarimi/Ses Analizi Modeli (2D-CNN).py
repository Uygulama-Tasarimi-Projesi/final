import tensorflow as tf
from tensorflow.keras import layers, models

def ses_analizi_cnn_modeli_olustur():
    # Mel-Spektrogramlar 128x128x1 boyutuna çekilir 
    input_shape = (128, 128, 1)
    model = models.Sequential(name="Ses_Analizi_2D_CNN_Modeli")

    # Görsel örüntüleri yakalamak için 32 filtre ve 3x3 kernel kullanılmıştır 
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name="Convolution_1"))
     
    # Boyut indirgeme ve en baskın özellikleri seçmek için 2x2 Max Pooling 
    model.add(layers.MaxPooling2D((2, 2), name="Pooling_1"))

    # Katman derinliği 64 filtreye çıkarılarak daha karmaşık akustik özellikler hedeflenir 
    model.add(layers.Conv2D(64, (3, 3), activation='relu', name="Convolution_2"))
    
    model.add(layers.MaxPooling2D((2, 2), name="Pooling_2"))

    # Daha derin duygu karakteristiklerinin yakalanması 
    model.add(layers.Conv2D(64, (3, 3), activation='relu', name="Convolution_3"))

    model.summary()
    
    return model

if __name__ == "__main__":
    model = ses_analizi_cnn_modeli_olustur()