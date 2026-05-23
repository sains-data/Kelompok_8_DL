# Prediksi Durasi Tidur Berdasarkan Kebiasaan Harian Menggunakan Multi-Layer Perceptron

## 📌 Deskripsi Project
Project ini bertujuan untuk memprediksi durasi tidur seseorang berdasarkan kebiasaan harian menggunakan algoritma Multi-Layer Perceptron (MLP). 

Penelitian dilakukan untuk memahami bagaimana faktor gaya hidup seperti screen time, frekuensi olahraga, konsumsi minuman tertentu, serta kebiasaan sehari-hari lainnya mempengaruhi durasi tidur mahasiswa.

Model dibangun menggunakan pendekatan machine learning dengan tahapan preprocessing data, normalisasi, pelatihan model, hyperparameter tuning, hingga evaluasi performa model.

---

## 🎯 Tujuan Project
- Menganalisis pengaruh kebiasaan harian terhadap durasi tidur
- Membangun model prediksi durasi tidur menggunakan algoritma MLP
- Mengevaluasi performa model menggunakan metrik MAE dan MSE
- Mengembangkan sistem prediksi sederhana berbasis machine learning

---

## 📊 Dataset
Dataset diperoleh melalui penyebaran kuesioner Google Form kepada mahasiswa aktif Program Studi Sains Data ITERA angkatan 2021–2025.

🔗 Link Dataset:  
[Dataset Excel](https://github.com/USERNAME/NAMA-REPO/blob/main/Data_Response_Gform_Deep_Learning%20(1).xlsx)

### Fitur Dataset
- Usia
- Jenis Kelamin
- Frekuensi Makan
- Screen Time
- Blue Light Filter
- Posisi Tidur
- Frekuensi Olahraga
- Kebiasaan Rokok/Alkohol
- Jenis Minuman yang Sering Dikonsumsi

### Target
- Durasi Tidur (Jam)

---

## 🛠 Tools dan Teknologi
- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Jupyter Notebook

---

## ⚙️ Metodologi
Tahapan yang dilakukan pada project ini meliputi:

1. Data Cleaning
2. Encoding Data Kategorik
3. Normalisasi Menggunakan Min-Max Scaling
4. Pembagian Data Train dan Test (80:20)
5. Pemodelan Multi-Layer Perceptron
6. Hyperparameter Tuning
7. Evaluasi Model Menggunakan MAE dan MSE

---

## 🧠 Arsitektur Model
Model Multi-Layer Perceptron yang digunakan terdiri dari:
- Input Layer
- Hidden Layer dengan 64 neuron
- Fungsi Aktivasi Tanh
- Optimizer Adam

---

## 📈 Hasil Evaluasi
Model menunjukkan performa yang stabil pada data train maupun test dengan hasil:

- MAE sekitar 0.49–0.50 jam
- MSE sekitar 0.44–0.45 jam²

Hasil evaluasi menunjukkan bahwa model mampu melakukan generalisasi dengan baik tanpa overfitting yang signifikan.

---

## 📷 Visualisasi

### Perbandingan Nilai Aktual dan Prediksi
![Aktual vs Prediksi](images/traintest.jpg)

---

## 🚀 Deployment
Project ini juga dilengkapi dengan deployment website sederhana yang memungkinkan pengguna memasukkan kebiasaan harian untuk mendapatkan prediksi durasi tidur secara interaktif.

---

## 📂 Struktur Project

```text
sleep-duration-prediction-mlp/
│
├── data/
├── images/
├── notebook/
├── app.py
├── model.pkl
├── requirements.txt
└── README.md
```

---

## 👩‍💻 Author
- Elia Meylani Simanjuntak
- Rut Junita Sari Siburian
- Rafa Aqilla Jungjunan

---
