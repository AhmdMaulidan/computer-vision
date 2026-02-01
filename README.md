# 🖥️ Computer Vision & Machine Learning Web Application

Aplikasi web interaktif untuk pembelajaran dan demonstrasi algoritma **Computer Vision** dan **Machine Learning**. Dibangun menggunakan Flask dengan antarmuka modern dan fitur visualisasi yang komprehensif.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.2-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)

---

## 📋 Daftar Isi

- [Fitur](#-fitur)
- [Teknologi](#-teknologi)
- [Struktur Project](#-struktur-project)
- [Instalasi](#-instalasi)
- [Menjalankan Aplikasi](#-menjalankan-aplikasi)
- [Penggunaan](#-penggunaan)
- [Dataset](#-dataset)
- [Screenshot](#-screenshot)
- [Lisensi](#-lisensi)

---

![image alt](https://github.com/AhmdMaulidan/computer-vision/blob/3bd5057656059a5d58c764188eea55739c94ad2e/Ficture%20project-5.png)

## ✨ Fitur

### 1. **Gray-Level Co-occurrence Matrix (GLCM)**

Analisis tekstur citra menggunakan metode GLCM untuk ekstraksi fitur:

- Upload gambar untuk dianalisis
- Parameter yang dapat dikustomisasi (angle, distance, window size)
- Menghasilkan feature maps untuk:
  - **Contrast** - Mengukur variasi intensitas lokal
  - **Correlation** - Mengukur ketergantungan linear antar piksel
  - **Energy** - Mengukur keseragaman tekstur
  - **Homogeneity** - Mengukur kedekatan distribusi GLCM ke diagonal

### 2. **K-Nearest Neighbors (KNN)**

Klasifikasi dan prediksi diabetes menggunakan algoritma KNN:

- Upload dataset CSV (Pima Indians Diabetes)
- Kustomisasi nilai K
- Visualisasi hasil:
  - Confusion Matrix
  - Classification Report
  - Accuracy Score
- Prediksi interaktif dengan input data pasien

### 3. **Naive Bayes**

Prediksi perilaku pembelian customer:

- Upload dataset Customer Behaviour
- Training model Gaussian Naive Bayes
- Evaluasi performa model
- Prediksi berdasarkan Gender, Age, dan Estimated Salary

### 4. **Decision Tree**

Klasifikasi data dengan visualisasi pohon keputusan:

- Upload dataset CSV dengan format fleksibel
- Visualisasi pohon keputusan (Decision Tree)
- Decision Boundary visualization
- Support untuk dataset dengan multiple classes
- Form prediksi dinamis berdasarkan kolom dataset

---

## 🛠️ Teknologi

| Kategori             | Teknologi                            |
| -------------------- | ------------------------------------ |
| **Backend**          | Python 3.9+, Flask, Flask-SQLAlchemy |
| **Machine Learning** | scikit-learn, scikit-image           |
| **Data Processing**  | NumPy, Pandas                        |
| **Image Processing** | OpenCV, Matplotlib                   |
| **Database**         | SQLite                               |
| **Frontend**         | HTML5, CSS3, Jinja2 Templates        |

---

## 📁 Struktur Project

```
web_visikomputer/
├── app.py                 # Main Flask application
├── tasks.db               # SQLite database
├── README.md              # Project documentation
│
├── dataset/               # Sample datasets
│   ├── K-NN/
│   │   └── diabetes.csv
│   ├── Naive Bayes/
│   │   └── Customer_Behaviour.csv
│   └── Decision Tree/
│       └── PlayTennis.csv
│
├── static/
│   ├── assets/            # Static images & thumbnails
│   ├── style.css          # Main stylesheet
│   └── uploads/           # Uploaded files & generated images
│
├── templates/
│   ├── base.html          # Base template
│   ├── index.html         # Homepage
│   ├── tugas.html         # Task list page
│   ├── task_detail.html   # GLCM detail page
│   ├── knn_detail.html    # KNN detail page
│   ├── naive_bayes_detail.html    # Naive Bayes detail page
│   └── decision_tree_detail.html  # Decision Tree detail page
│
└── venv/                  # Python virtual environment
```

---

## 🚀 Instalasi

### Prasyarat

- Python 3.9 atau lebih baru
- pip (Python package manager)

### Langkah Instalasi

1. **Clone repository**

   ```bash
   git clone <repository-url>
   cd web_visikomputer
   ```

2. **Buat virtual environment**

   ```bash
   python -m venv venv
   ```

3. **Aktifkan virtual environment**

   **macOS/Linux:**

   ```bash
   source venv/bin/activate
   ```

   **Windows:**

   ```bash
   venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install flask flask-sqlalchemy pandas numpy opencv-python matplotlib scikit-learn scikit-image
   ```

---

## ▶️ Menjalankan Aplikasi

1. **Aktifkan virtual environment** (jika belum aktif)

   ```bash
   source venv/bin/activate  # macOS/Linux
   # atau
   venv\Scripts\activate     # Windows
   ```

2. **Jalankan aplikasi**

   ```bash
   python app.py
   ```

3. **Buka browser dan akses**
   ```
   http://127.0.0.1:5001
   ```

---

## 📖 Penggunaan

### GLCM - Analisis Tekstur

1. Pilih menu **Gray-Level Co-occurrence Matrix**
2. Upload gambar yang ingin dianalisis
3. Atur parameter:
   - **Angle**: Sudut analisis (0°, 45°, 90°, 135°)
   - **Distance**: Jarak antar piksel
   - **Window Size**: Ukuran jendela (harus ganjil)
4. Klik **Analyze** untuk melihat feature maps

### KNN - Prediksi Diabetes

1. Pilih menu **K-Nearest Neighbors**
2. Upload dataset `diabetes.csv` (tersedia di folder `dataset/K-NN/`)
3. Tentukan nilai K
4. Lihat hasil evaluasi model
5. Masukkan data pasien untuk prediksi

### Naive Bayes - Prediksi Pembelian

1. Pilih menu **Naive Bayes**
2. Upload dataset `Customer_Behaviour.csv` (tersedia di folder `dataset/Naive Bayes/`)
3. Lihat hasil training model
4. Masukkan data customer untuk prediksi

### Decision Tree - Klasifikasi

1. Pilih menu **Decision Tree**
2. Upload dataset CSV (kolom terakhir = target)
3. Lihat visualisasi pohon keputusan
4. Gunakan form prediksi untuk klasifikasi data baru

---

## 📊 Dataset

Project ini menyediakan sample dataset:

| Dataset                  | Lokasi                   | Deskripsi                          |
| ------------------------ | ------------------------ | ---------------------------------- |
| `diabetes.csv`           | `dataset/K-NN/`          | Pima Indians Diabetes Dataset      |
| `Customer_Behaviour.csv` | `dataset/Naive Bayes/`   | Data perilaku pembelian customer   |
| `PlayTennis.csv`         | `dataset/Decision Tree/` | Dataset klasik untuk Decision Tree |

### Format Dataset

**diabetes.csv:**

```
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
```

**Customer_Behaviour.csv:**

```
Gender, Age, EstimatedSalary, Purchased
```

**PlayTennis.csv:**

```
Outlook, Temperature, Humidity, Wind, PlayTennis
```

---

## 📸 Screenshot

_Coming soon_

---

## 👨‍💻 Author

Dibuat untuk keperluan mata kuliah **Visi Komputer** - Semester 5

---

## 📄 Lisensi

Project ini dibuat untuk keperluan akademik.

---

## 🤝 Kontribusi

Kontribusi selalu diterima! Silakan buat pull request atau laporkan issues jika menemukan bug.
