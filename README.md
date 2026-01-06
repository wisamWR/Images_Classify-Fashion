# Fashion MNIST AI Classifier 🧥👟

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

Aplikasi Deep Learning untuk mengklasifikasikan gambar pakaian fashion ke dalam 10 kategori menggunakan **Convolutional Neural Network (CNN)**.

## 📋 Tentang Proyek
Proyek ini bertujuan untuk mendemonstrasikan bagaimana Kecerdasan Buatan (AI) dapat digunakan dalam industri fashion dan e-commerce untuk otomatisasi kategorisasi produk.

Model dilatih menggunakan dataset **Fashion MNIST** yang terdiri dari 70.000 gambar grayscale (28x28 piksel).

**Akurasi Model:** ~91% (Advanced CNN dengan Data Augmentation).

## 🚀 Demo Aplikasi
Anda dapat mencoba aplikasi ini secara langsung di Streamlit Cloud:
[Link Aplikasi Anda Nanti Disini]

## 🛠️ Instalasi & Jalankan Lokal

Jika Anda ingin menjalankan proyek ini di komputer Anda sendiri:

1.  **Clone Repository**
    ```bash
    git clone https://github.com/username-anda/fashion-mnist-app.git
    cd fashion-mnist-app
    ```

2.  **Buat Virtual Environment (Opsional tapi Disarankan)**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependensi**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Aplikasi**
    ```bash
    streamlit run app.py
    ```

## 🧠 Arsitektur Model
Kami membandingkan beberapa arsitektur sebelum memilih yang terbaik:
*   **Baseline ANN**: Akurasi ~82%
*   **Simple CNN**: Akurasi ~88%
*   **Advanced CNN (Dipilih)**: Menggunakan 3 blok konvolusi, Batch Normalization, Dropout, dan Global Average Pooling. Akurasi mencapai **>90%**.

## 📂 Struktur Folder
*   `app.py`: Kode utama aplikasi web (Streamlit).
*   `train_model.py`: Script untuk melatih ulang model.
*   `fashion_model.h5`: File model yang sudah dilatih (disimpan).
*   `requirements.txt`: Daftar pustaka Python yang dibutuhkan.
*   `Fashion_MNIST_Project.ipynb`: Notebook eksperimen lengkap.

## 🤝 Kontribusi
Silakan buka **Issue** atau **Pull Request** jika Anda memiliki saran perbaikan!

---
*Dibuat untuk tugas Deep Learning Coursera.*
