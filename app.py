import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# =================== LOAD MODEL ===================
MODEL_PATH = "model_sampah.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =================== JUDUL APLIKASI ===================
st.title("🗑️ Klasifikasi Sampah Organik vs Anorganik")
st.write("Upload gambar sampah, dan sistem akan memprediksi jenisnya.")

# =================== CONTOH GAMBAR ===================
if os.path.exists("examples/organic_sample.jpg"):
    st.image("examples/organic_sample.jpg", caption="Contoh Sampah Organik", width=200)
if os.path.exists("examples/anorganic_sample.jpg"):
    st.image("examples/anorganic_sample.jpg", caption="Contoh Sampah Anorganik", width=200)

# =================== UPLOAD GAMBAR ===================
uploaded_file = st.file_uploader("📤 Upload gambar sampah (.jpg/.png)", type=["jpg", "jpeg", "png"])

# =================== FUNGSI PREDIKSI ===================
def predict(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "Organik" if prediction < 0.5 else "Anorganik"

# =================== TAMPILKAN HASIL ===================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Gambar yang diupload", use_column_width=True)
    label = predict(img)
    st.success(f"Hasil Prediksi: **{label}**")
