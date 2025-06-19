import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# ====== Langkah 1: Buat struktur folder jika belum ada ======
required_folders = [
    "dataset/train/organik",
    "dataset/train/anorganik",
    "dataset/validation/organik",
    "dataset/validation/anorganik"
]

for folder in required_folders:
    os.makedirs(folder, exist_ok=True)

# ====== Langkah 2: Cek isi folder ======
def cek_folder(path):
    if not os.path.exists(path):
        print(f"[❌] Folder tidak ditemukan: {path}")
    elif len(os.listdir(path)) == 0:
        print(f"[⚠️] Folder kosong: {path}")
    else:
        print(f"[✅] {path} berisi {len(os.listdir(path))} file:")
        print("     ", os.listdir(path))

for folder in required_folders:
    cek_folder(folder)

# ====== Langkah 3: Preprocessing data ======
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

val_data = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

# ====== Langkah 4: Buat model CNN ======
model = Sequential([
    tf.keras.Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # binary classification
])

# ====== Langkah 5: Compile dan Latih ======
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=10)

# ====== Langkah 6: Simpan model ======
model.save("model_sampah.h5")
print("\n[✅] Model berhasil disimpan sebagai model_sampah.h5")
