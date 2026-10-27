# Audit Integritas Data HQC-2026, Mitigasi Target Leakage, dan Evaluasi Komparatif QSVC vs Classical Baseline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-Quantum%20Computing-6929C4.svg)](https://qiskit.org/)
[![ITS](https://img.shields.io/badge/ITS-Teknologi%20Informasi-003366.svg)](https://www.its.ac.id/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Proyek riset defensif dan komputasi ilmiah untuk mata kuliah **Quantum Computing untuk Cybersecurity**, Departemen Teknologi Informasi, Institut Teknologi Sepuluh Nopember (ITS).

---

## 👥 Tim Peneliti / Anggota Kelompok

| No. | Nama Mahasiswa | NRP |
| :---: | :--- | :---: |
| 1 | **Rafael Gunawan** | 5027231019 |
| 2 | **Muhammad Nafi Firdaus** | 5027231045 |
| 3 | **Abhirama Triadyatama Hermawan** | 5027231061 |
| 4 | **Danendra Fidel Khansa** | 5027231063 |
| 5 | **Tio Axellino Irin** | 5027231065 |

**Dosen Pengampu / Acuan Modul:** Dr. Hatma Suryotrisongko (Modul Ajar Bab 6: *HQC-2026 untuk Quantum Machine Learning*, 2026).

---

## 🎯 Ringkasan Eksekutif & Temuan Kritis (Zero False Claims)

Riset ini mengusung prinsip transparansi ilmiah penuh (*Zero False Claims*) dalam mengevaluasi dataset **HQC-2026** (*A Hybrid Quantum-Classical Network Intrusion Detection Dataset*, Mendeley Data DOI: [10.17632/68rc9crgkj.1](https://doi.org/10.17632/68rc9crgkj.1)):

1. **Audit vs Halaman Publik:**
   - Membuktikan secara empiris kesenjangan antara klaim publik Mendeley (klaim 14,69 juta baris, seimbang via SMOTE) vs artefak fisik Parquet aktual (233.844 baris dengan ketimpangan ekstrem: Benign mendominasi 75,99%).
2. **Eksploitasi & Pembuktian Target Leakage:**
   - Menemukan redundansi deterministik: `sifted_key_rate = 1 - qber` dan `_delta = qber - 0.01`.
   - Mengungkap *target leakage* fatal pada fitur `is_anomalous_qkd == 1` yang mencakup **100% sampel serangan**. Tanpa pembersihan fitur ini, model akan mencapai metrik artifisial 100% akurasi semu.
3. **Benchmarking QSVC vs Classical:**
   - Melakukan mitigasi kebocoran data dengan isolasi fitur kuantum autentik.
   - Menguji keunggulan komparatif Quantum Support Vector Classifier (QSVC) berbasis quantum kernel/feature maps terhadap baseline klasik (Random Forest, SVM, LightGBM).

---

## 📁 Struktur Direktori Repository

```text
.
├── QML_Cybersecurity_HQC2026.ipynb                # Notebook riset utama (End-to-End)
├── Bab_6_Dataset_HQC_2026_untuk_QML_ITS.pdf       # Modul referensi Bab 6
├── Presentasi_Bab_6_Dataset_HQC_2026_untuk_QML_ITS.pdf # Slide presentasi riset
├── requirements_frozen.txt                        # Freeze seluruh environment eksperimen
├── data/
│   └── README.md                                  # Panduan pengunduhan dataset fisik
├── scratch/
│   ├── build_full_notebook.py                     # Generator otomasi sel notebook
│   ├── run_experiments.py                         # Skrip eksekusi eksperimen
│   ├── run_temporal_experiment.py                 # Eksperimen validasi temporal
│   ├── experiment_results.json                    # Log metrik performa model
│   └── experiment_summary.csv                     # Rekapitulasi tabel hasil eksperimen
├── .gitignore                                     # Filter proteksi file >100MB & venv
└── README.md                                      # Dokumentasi utama proyek
```

---

## 🚀 Panduan Menjalankan Proyek (Reproducibility)

### 1. Kloning Repository
```bash
git clone https://github.com/Ludw1gd/QML-Cybersecurity-HQC2026.git
cd QML-Cybersecurity-HQC2026
```

### 2. Setup Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Instalasi Dependensi
```bash
pip install --upgrade pip
pip install -r requirements_frozen.txt
```

### 4. Persiapan Dataset
Unduh file dataset Parquet dari [Mendeley Data (DOI: 10.17632/68rc9crgkj.1)](https://doi.org/10.17632/68rc9crgkj.1) dan tempatkan pada folder `data/` sesuai petunjuk pada [`data/README.md`](data/README.md).

### 5. Menjalankan Notebook
```bash
jupyter lab QML_Cybersecurity_HQC2026.ipynb
```

---

## ⚖️ Etika Ilmiah & Sitasi

Proyek ini dibuat untuk tujuan akademik dan investigasi ilmiah integritas data. Jika menggunakan temuan atau kode dari repository ini, mohon cantumkan atribusi:

```bibtex
@misc{qml_cybersecurity_hqc2026,
  author = {Gunawan, Rafael and Firdaus, Muhammad Nafi and Hermawan, Abhirama Triadyatama and Khansa, Danendra Fidel and Irin, Tio Axellino},
  title = {Audit Integritas Data HQC-2026, Mitigasi Target Leakage, dan Evaluasi Komparatif QSVC vs Classical Baseline},
  year = {2026},
  institution = {Departemen Teknologi Informasi, Institut Teknologi Sepuluh Nopember (ITS)}
}
```
