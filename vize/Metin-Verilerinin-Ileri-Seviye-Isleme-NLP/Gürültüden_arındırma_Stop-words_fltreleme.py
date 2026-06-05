import pandas as pd
import re
import jpype
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ZEMBEREK BAŞLAT
ZEMBEREK_PATH = "zemberek-full.jar"

if not jpype.isJVMStarted():
    jpype.startJVM(jpype.getDefaultJVMPath(), "-Dfile.encoding=UTF8", "-cp", ZEMBEREK_PATH)

TurkishMorphology = jpype.JClass("zemberek.morphology.TurkishMorphology")
morphology = TurkishMorphology.createWithDefaults()

print("--- VERİ YÜKLENİYOR ---")
df = pd.read_csv("train_metin.csv")

# STOPWORDS (Etkisiz Kelimeler)
with open("stop_words.txt", "r", encoding="utf-8") as f:
    stop_words = set(f.read().split())

# GÜVENLİ MORFOLOJİK ANALİZ (Ön İşleme Fonksiyonu)
# Sadece çekim eklerini temizle, türetme eklerine dokunma.
# Lemma al ama sadece lemma kelimeye çok yakınsa kabul et
# Uzunluk farkı çoksa surface formu korur
def metin_on_isleme(metin):

    if not isinstance(metin, str):
        return ""

    # Gürültü temizleme
    metin = re.sub(r'https?://\S+|www\.\S+', '', metin)
    metin = re.sub(r'\bRT\b', '', metin)
    metin = re.sub(r'@\w+', '', metin)

    # Türkçe küçük harf
    metin = metin.replace("I", "ı").replace("İ", "i").lower()

    # Noktalama temizleme
    metin = re.sub(r'[^a-zçğıöşü\s]', ' ', metin)
    metin = re.sub(r'\s+', ' ', metin).strip()

    if not metin:
        return ""

    temiz_kelimeler = []

    try:
        analysis = morphology.analyzeSentence(metin)
        results = morphology.disambiguate(metin, analysis).bestAnalysis()

        for result in results:

            original = str(result.surfaceForm())
            lemma = str(result.getDictionaryItem().lemma)

            if original in stop_words:
                continue

            # ANLAM KORUMA KURALI
            # Sadece çekim eki varsa lemma kullan
            # Uzunluk farkı çoksa surface formu koru
            length_diff = len(original) - len(lemma)

            if lemma != "UNK" and 0 <= length_diff <= 3:
                temiz_kelimeler.append(lemma)
            else:
                temiz_kelimeler.append(original)

    except:
        for kelime in metin.split():
            if kelime not in stop_words:
                temiz_kelimeler.append(kelime)

    return " ".join(temiz_kelimeler)

# TEMİZLİĞİ UYGULA
print("\n--- METİNLER İŞLENİYOR (Ön İşleme Başladı) ---")
df["Temiz_Text"] = df["Text"].apply(metin_on_isleme)
df = df[df["Temiz_Text"].str.strip() != ""]

# Metinleri Sayılara Dönüştür
print("\n--- TOKENİZASYON UYGULANIYOR ---")
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df["Temiz_Text"])
sequences = tokenizer.texts_to_sequences(df["Temiz_Text"])

# Sabit Uzunluğa Getirme
print("\n--- PADDING UYGULANIYOR ---")
MAX_UZUNLUK = 50
X_padded = pad_sequences(sequences, maxlen=MAX_UZUNLUK, padding="post")

df.to_csv("temiz_train_metin.csv", index=False, encoding="utf-8")

print("\n--- İŞLEM TAMAMLANDI ---")
print("NLP ön işleme başarıyla tamamlandı.")
print(f"Model giriş matrisi boyutu: {X_padded.shape}")

jpype.shutdownJVM()