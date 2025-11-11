🧠 Sistem Temu Kembali Informasi (UTS STKI - A11.2023.15189)

Proyek ini merupakan implementasi Sistem Temu Kembali Informasi (Information Retrieval System) menggunakan bahasa Python, yang mencakup Boolean Retrieval Model dan Vector Space Model (VSM) dengan evaluasi metrik IR standar (Precision, Recall, F1, MAP, dan nDCG).

📂 **Struktur Folder:**
```plaintext
stki-uts/
├── app/
│   └── main.py
├── src/
│   ├── preprocess.py
│   ├── boolean_ir.py
│   ├── vsm_ir.py
│   ├── eval.py
│   └── search.py
├── data/
│   ├── RPS Kriptografi.txt
│   ├── RPS Sistem Informasi.txt
│   ├── RPS Sistem Temu Kembali Informasi.txt
│   ├── RPS Sistem Terdistribusi.txt
│   ├── RPS Manajemen Proyek Teknologi Informasi.txt
│   └── stopwords.txt
├── data_processed/             # hasil preprocessing (otomatis dibuat)
├── notebooks/
│   └── UTS_STKI_A11_2023_15189.ipynb
├── reports
│   ├── laporan.pdf
│   └── readme.md
└── requirements.txt

Langkah Menjalankan Program:
1️⃣ Instalasi Dependensi
Pastikan Python versi ≥3.8 sudah terinstal.
Kemudian jalankan perintah berikut di terminal (dalam root proyek):

pip install numpy scipy scikit-learn Sastrawi

2️⃣ Jalankan Menu Utama
Program utama dapat dijalankan melalui:

python app/main.py

Tampilan awal:

=== UTS STKI - MAIN MENU ===
1) Preprocess all documents (generate data_processed/)
2) Build indices (inverted index + incidence matrix)
3) Boolean query (interactive)
4) Build VSM (TF-IDF) and run example query
5) Interactive VSM search (top-K)
6) Run evaluation examples (Precision/Recall/F1/nDCG)
0) Exit

3️⃣ Langkah Eksekusi Tiap Soal
No	Tahap	File / Fungsi Utama	Output
1	Preprocessing (Soal 02)	src/preprocess.py	File hasil: data_processed/CLEAN_*.txt
2	Boolean Retrieval (Soal 03)	src/boolean_ir.py	Inverted Index, Incidence Matrix, Precision & Recall
3	Vector Space Model (Soal 04)	src/vsm_ir.py	TF-IDF Matrix, Cosine Similarity, MAP
4	Evaluasi Metrik (Soal 05)	src/eval.py	Precision, Recall, F1, dan nDCG
5	Search Engine Interaktif	src/search.py	CLI pencarian query bebas
6	Main Integration	app/main.py	Menu terintegrasi seluruh soal

🧩 Asumsi & Catatan Teknis

1. Dataset
    - Dokumen berupa file .txt berisi Rencana Pembelajaran Semester (RPS) dari berbagai mata kuliah.
    - Semua file ditempatkan di folder data/.
    - Nama file mengikuti format RPS <Nama Mata Kuliah>.txt.

2. Stopwords
    - Daftar kata umum (stopwords) Bahasa Indonesia disimpan di data/stopwords.txt.
    - Jika file tidak ditemukan, sistem akan menampilkan peringatan dan melanjutkan dengan stopword kosong.

3. Preprocessing
    - Proses: case folding → tokenization → stopword removal → stemming (menggunakan library Sastrawi).
    - Hasil setiap dokumen disimpan di data_processed/ dalam format:
    CLEAN_RPS <Nama Mata Kuliah>.txt

4. Boolean Retrieval
    - Mendukung operator AND, OR, NOT.
    - Query diasumsikan sudah dalam bentuk stemmed (misalnya sistem AND distribusi).

5. Vector Space Model (VSM)
    - Menggunakan bobot TF-IDF dan metrik Cosine Similarity.
    - Ranking dokumen ditampilkan dengan top-K hasil terbaik.
    - Evaluasi otomatis menghitung Precision@K dan Mean Average Precision (MAP).

6. Evaluasi
    - File eval.py mengimplementasikan metrik:
        - Precision, Recall, F1-Score
        - DCG / nDCG untuk menilai peringkat hasil pencarian
    - Contoh hasil dapat dijalankan dengan:
        python src/eval.py

7. Main Program
    - File main.py mengintegrasikan seluruh modul ke dalam satu antarmuka CLI.
    - Pengguna dapat menjalankan semua tahapan dari preprocessing sampai evaluasi dari satu tempat.


📊 Contoh Output (Ringkas)

Hasil Preprocessing:
    [OK] CLEAN_RPS Sistem Informasi.txt disimpan (1230 token)
    ✅ Semua dokumen tersimpan di folder: data_processed/

Hasil Boolean Retrieval:
    QUERY: 'kriptografi AND keamanan'
    > Retrieved Docs: ['D4']
    > Precision=1.0, Recall=1.0, F1=1.0

Hasil VSM:
    SOAL 04: VECTOR SPACE MODEL (VSM)
    MAP: 0.6389


🧾 Lisensi & Kredit
    Proyek ini dikembangkan untuk keperluan UTS Mata Kuliah Sistem Temu Kembali Informasi (STKI)
    Program Studi Teknik Informatika, Fakultas Ilmu Komputer
    Universitas Dian Nuswantoro (UDINUS)
    🧑‍🎓 Dikerjakan oleh: Aditya Rendy Setyawan – A11.2023.15189