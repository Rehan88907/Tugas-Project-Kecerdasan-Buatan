# 🗑️ Smart Waste Classifier

Aplikasi klasifikasi gambar sampah **organik** dan **anorganik** berbasis **Convolutional Neural Network (CNN)** dan **Streamlit**.

## 📸 Contoh

![Interface](examples/preview.png)

---

## 🚀 Fitur
- Upload gambar sampah
- Prediksi otomatis (organik atau anorganik)
- Tampilan web interaktif via Streamlit

---

## 🗂️ Struktur Folder



---

## 📦 Instalasi & Menjalankan

### 1. Clone Repo (atau download ZIP)

```bash
git clone https://github.com/username/smart-waste-classifier.git
cd smart-waste-classifier

Sistem Smart Waste/
├── app.py # Streamlit app
├── model.py # Training model CNN
├── model_sampah.h5 # Model hasil training
├── dataset/ # Gambar training & validasi
│ ├── train/
│ │ ├── organik/
│ │ └── anorganik/
│ └── validation/
│ ├── organik/
│ └── anorganik/
├── examples/ # Gambar contoh untuk antarmuka
│ ├── organic.jpg
│ └── anorganic.jpg
├── requirements.txt # Daftar dependensi
└── README.md # Dokumentasi proyek
