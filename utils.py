import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image(image, target_size=(150, 150)):
    """Mempersiapkan gambar untuk prediksi"""
    if image.mode != "RGB":
        image = image.convert("RGB")  # Mengubah ke RGB jika bukan RGB
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalisasi piksel
    image = np.expand_dims(image, axis=0)  # Ubah jadi bentuk (1, 150, 150, 3)
    return image

def load_model(model_path='smart_waste_cnn_model.h5'):
    """Memuat model CNN yang sudah dilatih"""
    return tf.keras.models.load_model(model_path)

