# Dataset Instruction (HQC-2026)

Dataset fisik tidak dilacak dalam repository Git ini karena ukurannya melebihi batas upload GitHub (>100MB per file).

### Sumber Dataset Resmi
- **Dataset Utama:** *A Hybrid Quantum-Classical Network Intrusion Detection Dataset* (HQC-2026)
- **Penyedia:** Mendeley Data
- **DOI:** [10.17632/68rc9crgkj.1](https://doi.org/10.17632/68rc9crgkj.1) (Version 1, diterbitkan 9 Januari 2026)
- **Modul Acuan:** Modul Ajar Bab 6: *HQC-2026 untuk Quantum Machine Learning* (Dr. Hatma Suryotrisongko, Departemen Teknologi Informasi ITS, 2026)

### Struktur File yang Diharapkan
Letakkan file dataset berikut di dalam direktori `data/`:
```text
data/
├── HQC20_Super_Enriched_Dataset_v4.parquet
├── SUPER_ENRICHED_hybrid_dataset_v3.parquet
└── final_hybrid_cybersecurity_dataset.parquet
```
Semua file `.parquet` dan `.zip` diabaikan secara otomatis oleh `.gitignore`.
