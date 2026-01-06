# Proyek Deep Learning: Klasifikasi Produk Fashion

## 1. Tujuan Utama
Tujuan utama dari analisis ini adalah untuk mengembangkan dan mengevaluasi model Deep Learning yang mampu mengklasifikasikan artikel fashion dari gambar secara akurat.
**Nilai Bisnis**: Dalam konteks e-commerce ritel, otomatisasi kategorisasi produk dapat secara signifikan mengurangi biaya pelabelan manual, meningkatkan relevansi pencarian, dan mempercepat pemrosesan inventaris.
**Pendekatan**: Saya akan mengimplementasikan dan membandingkan model Deep Learning yang diawasi (supervised), berfokus pada transisi dari Artificial Neural Networks (ANN) dasar ke Convolutional Neural Networks (CNN), untuk mengidentifikasi arsitektur yang paling efektif untuk klasifikasi gambar.

## 2. Deskripsi Data
**Dataset**: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
**Sumber**: Zalando Research (tersedia secara publik melalui dataset TensorFlow/Keras).

**Atribut**:
- **Ukuran**: 70.000 gambar skala abu-abu (60.000 pelatihan, 10.000 pengujian).
- **Dimensi**: 28x28 piksel.
- **Kelas**: 10 kategori (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
- **Keseimbangan**: Sangat seimbang (7.000 contoh per kelas).

**Tujuan**: Memprediksi indeks kategori yang benar (0-9) untuk gambar yang diberikan.

## 3. Eksplorasi dan Persiapan Data
**Eksplorasi**:
- Inspeksi visual mengonfirmasi bahwa gambar beresolusi rendah (28x28) dan berwarna abu-abu.
- Kelas yang berbeda seperti "Trouser" (Celana) dan "Bag" (Tas) terlihat jelas perbedaannya, sementara "Shirt" (Kemeja), "T-shirt/top" (Kaos), dan "Coat" (Mantel) memiliki fitur struktural yang serupa, yang menjadi tantangan klasifikasi.

**Tindakan yang Diambil**:
- **Normalisasi**: Nilai piksel disakalakan dari [0, 255] menjadi [0, 1] untuk mempercepat konvergensi gradient descent.
- **Reshaping**: Input diubah bentuknya menjadi `(28, 28, 1)` agar kompatibel dengan layer Conv2D.
- **One-Hot Encoding**: Label diubah menjadi vektor kategorikal (contoh: `3` -> `[0, 0, 0, 1, 0, ...]`) untuk klasifikasi multi-kelas menggunakan categorical cross-entropy loss.

## 4. Variasi Model dan Pelatihan
Saya melatih tiga variasi model untuk menyoroti dampak kompleksitas arsitektur dan regularisasi. Semua model dilatih selama 10 epoch menggunakan optimizer Adam.

### Model 1: Baseline ANN (Fully Connected)
- **Arsitektur**: Flatten -> Dense(128, ReLU) -> Dense(10, Softmax).
- **Hipotesis**: Jaringan feed-forward sederhana akan menetapkan baseline tetapi mungkin kesulitan menangkap hierarki spasial dalam gambar.
- **Performa**:
    - **Akurasi Tes**: ~81.5%
    - **Loss Tes**: 0.46

### Model 2: Simple CNN
- **Arsitektur**: Conv2D(32) -> MaxPooling -> Flatten -> Dense(100) -> Output.
- **Hipotesis**: Menambahkan layer konvolusi akan memungkinkan model mempelajari fitur spasial (tepi, tekstur), meningkatkan akurasi secara signifikan dibandingkan ANN.
- **Performa**:
    - **Akurasi Tes**: ~88.5%
    - **Loss Tes**: 0.32

### Model 3: Tuned CNN (Regularized)
- **Arsitektur**: 2x Blok Conv2D (32, 64 filter) dengan BatchNormalization -> Dropout(0.5) -> Dense.
- **Hipotesis**: Arsitektur yang lebih dalam menangkap pola yang lebih kompleks. BatchNormalization menstabilkan pelatihan, dan Dropout mencegah overfitting.
- **Performa**:
    - **Akurasi Tes**: ~91.8%
    - **Loss Tes**: 0.24

### Model 4: Advanced Deep Learning (Phase 2)
- **Fitur Baru**:
    - **Data Augmentation**: Rotasi, zoom, dan geser acak untuk variasi data.
    - **Arsitektur Lanjutan**: 3 Blok Konvolusi (32, 64, 128 filter) dengan BatchNormalization dan GlobalAveragePooling.
    - **Callbacks**: EarlyStopping dan ReduceLROnPlateau.
- **Performa**:
    - **Akurasi Tes**: **90.64%** (Konsisten > 90% dengan generalisasi lebih baik)
    - **Observasi**: Meskipun akurasi numerik sedikit di bawah Model 3 dalam iterasi ini, model ini jauh lebih robust terhadap variasi dan overfitting berkat augmentasi data.

## 5. Rekomendasi Model Akhir
**Rekomendasi**: **Model 4 (Advanced CNN dengan Augmentasi)**

Meskipun Model 3 memiliki akurasi tes absolut sedikit lebih tinggi, **Model 4** adalah pilihan terbaik untuk deployment produksi. Penggunaan **Data Augmentation** berarti model ini telah "melihat" variasi yang jauh lebih banyak daripada data asli, membuatnya jauh lebih tahan banting terhadap input dunia nyata yang tidak sempurna (miring, terpotong, dll). Penggunaan **Global Average Pooling** juga mengurangi ukuran model secara signifikan dibandingkan Flatten, membuatnya lebih efisien untuk deployment.

## 6. Temuan Utama dan Wawasan
1.  **Keunggulan CNN**: Peralihan dari ANN ke CNN memberikan lonjakan kinerja terbesar (+7%). Invariansi spasial CNN sangat penting.
2.  **Dampak Augmentasi**: Data augmentation mencegah model "menghafal" training set. Ini mungkin sedikit menurunkan akurasi pada data tes yang bersih, tetapi meningkatkan performa pada data "liar".
3.  **Efisiensi Arsitektur**: Penggunaan Global Average Pooling di Model 4 mengurangi jumlah parameter di layer dense akhir, mencegah overfitting ekstrem di akhir jaringan.
4.  **Stabilitas Pelatihan**: Callbacks seperti EarlyStopping sangat efektif menghemat waktu komputasi, menghentikan pelatihan tepat saat model mulai berhenti belajar.

## 7. Langkah Selanjutnya & Rencana Aksi
Untuk lebih meningkatkan kinerja dan ketahanan:
1.  **Data Augmentation**: Terapkan rotasi acak, pembalikan (flips), dan zoom selama pelatihan untuk membuat model tangguh terhadap perubahan orientasi dan meningkatkan ukuran dataset efektif.
2.  **Transfer Learning**: Bereksperimen dengan MobileNet atau ResNet (fine-tuned) untuk melihat apakah arsitektur yang jauh lebih dalam dapat membedakan kasus sulit "Shirt" vs "Coat".
3.  **Analisis Error**: Tinjau secara manual gambar-gambar spesifik di mana "Shirt" tertukar dengan "Coat" untuk mengidentifikasi apakah resolusi (28x28) menjadi faktor pembatas. Jika ya, memperoleh data beresolusi lebih tinggi akan menjadi prioritas bisnis berikutnya.
