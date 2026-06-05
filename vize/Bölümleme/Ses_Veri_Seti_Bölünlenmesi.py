import os
import shutil
import random

print("--- SES VERİ SETİ BÖLÜMLENDİRME BAŞLIYOR ---")
ravdess_path = r"archive_ravdess"
output_dir = r"ravdess_bolunmus"

# Eğitimde tutarlılık için rastgelelik sabiti
random.seed(42)

# AKTÖRLERİ AL VE KARIŞTIR
# Veri sızıntısını önlemek için bölümlendirme
# Dosya bazlı değil, aktör (konuşmacı) bazlı 
actors = [a for a in os.listdir(ravdess_path) if a.startswith("Actor")]
actors.sort()
random.shuffle(actors)

total = len(actors)

# %80 Train, %10 Validation, %10 Test oranında bölümlendirme
train_end = int(total * 0.8)
val_end = int(total * 0.9)

train_actors = actors[:train_end]
val_actors = actors[train_end:val_end]
test_actors = actors[val_end:]

print("\n--- AKTÖR DAĞILIMI ---")
print(f"Train Aktörleri ({len(train_actors)}): {train_actors}")
print(f"Validation Aktörleri ({len(val_actors)}): {val_actors}")
print(f"Test Aktörleri ({len(test_actors)}): {test_actors}")

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

def copy_files(actor_list, split_name):
    for actor in actor_list:
        actor_path = os.path.join(ravdess_path, actor)
        files = [f for f in os.listdir(actor_path) if f.endswith(".wav")]

        for file in files:
            src = os.path.join(actor_path, file)
            dst_dir = os.path.join(output_dir, split_name, actor)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src, dst_dir)

print("\n--- DOSYALAR KOPYALANIYOR ---")
print("Train (%80) klasörü oluşturuluyor...")
copy_files(train_actors, "train")

print("Validation (%10) klasörü oluşturuluyor...")
copy_files(val_actors, "val")

print("Test (%10) klasörü oluşturuluyor...")
copy_files(test_actors, "test")

print("\n--- İŞLEM TAMAMLANDI ---")
print("RAVDESS veri seti %80-%10-%10 oranında aktör bazlı olarak başarıyla bölündü.")
