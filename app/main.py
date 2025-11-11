import os
import sys
import time
from typing import List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data_processed")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from preprocess import initialize_preprocessing, get_processed_corpus, preprocess_query
except Exception as e:
    try:
        from preprocess import initialize_preprocessing, get_processed_corpus as _gpc, preprocess_query as _pq
        initialize_preprocessing = initialize_preprocessing
        get_processed_corpus = _gpc
        preprocess_query = _pq
    except Exception:
        print("Warning: Tidak dapat mengimpor fungsi dari preprocess.py. Pastikan file src/preprocess.py ada dan memiliki fungsi:")
        print("  - initialize_preprocessing()")
        print("  - get_processed_corpus()")
        print("  - preprocess_query(query, stemmer, stop_words)")
        print("Continuing; some features may not work.")
        initialize_preprocessing = None
        get_processed_corpus = None
        preprocess_query = None

try:
    from boolean_ir import build_inverted_index, build_incidence_matrix, boolean_retrieve, calculate_precision_recall
except Exception:
    try:
        from boolean_ir import build_inverted_index, build_incidence_matrix, boolean_retrieve, calculate_precision_recall
    except Exception:
        build_inverted_index = None
        build_incidence_matrix = None
        boolean_retrieve = None
        calculate_precision_recall = None
        print("Warning: boolean_ir.py import error. Boolean features may not work.")

try:
    from vsm_ir import calculate_tf_idf, query_to_tfidf_vector, rank_documents, load_processed_documents as vsm_load_processed
except Exception:
    try:
        from vsm_ir import calculate_tf_idf, query_to_tfidf_vector, rank_documents
        vsm_load_processed = None
    except Exception:
        calculate_tf_idf = None
        query_to_tfidf_vector = None
        rank_documents = None
        vsm_load_processed = None
        print("Warning: vsm_ir.py import error. VSM features may not work.")

try:
    from eval import evaluate_search_engine as eval_search
except Exception:
    try:
        from eval import evaluate as eval_search
    except Exception:
        eval_search = None
        print("Warning: eval.py import error. Evaluation features may not work.")

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

def run_preprocessing_and_save():
    """Menjalankan get_processed_corpus() lalu menyimpan ke data_processed/ sebagai CLEAN_*.txt."""
    if get_processed_corpus is None:
        print("Preprocess module tidak tersedia. Jalankan/memperbaiki src/preprocess.py")
        return

    print("Memulai preprocessing korpus...")
    start = time.time()
    processed = get_processed_corpus()
    elapsed = time.time() - start
    print(f"Preprocessing selesai dalam {elapsed:.2f} detik. Menyimpan hasil ke '{DATA_PROCESSED_DIR}' ...")

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    count = 0
    for doc_id, tokens in processed.items():
        out_name = f"CLEAN_{doc_id}"
        out_path = os.path.join(DATA_PROCESSED_DIR, out_name)
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(" ".join(tokens))
            count += 1
        except Exception as ex:
            print(f"  Gagal menulis {out_path}: {ex}")
    print(f"Selesai: {count} file ditulis ke data_processed/.")

def build_indices_from_processed():
    """Membaca files CLEAN_*.txt, membangun inverted index dan incidence matrix."""
    ensure_dirs()
    if not os.path.isdir(DATA_PROCESSED_DIR):
        print("Folder data_processed tidak ditemukan. Jalankan preprocessing dulu.")
        return

    file_list = sorted([f for f in os.listdir(DATA_PROCESSED_DIR) if f.startswith("CLEAN_") and f.endswith(".txt")])
    if not file_list:
        print("Tidak ada file CLEAN_*.txt di data_processed/. Jalankan preprocessing dulu.")
        return

    documents = {}
    doc_id_map = {}
    all_terms = set()
    for i, fn in enumerate(file_list):
        path = os.path.join(DATA_PROCESSED_DIR, fn)
        with open(path, "r", encoding="utf-8") as f:
            toks = f.read().split()
        doc_id = f"D{i+1}"
        documents[doc_id] = toks
        doc_id_map[doc_id] = fn.replace("CLEAN_", "").replace(".txt", "")
        all_terms.update(toks)
    vocabulary = sorted(list(all_terms))

    print(f"Loaded {len(documents)} documents, vocabulary size: {len(vocabulary)}")

    inverted = None
    incidence = None
    if build_inverted_index is not None:
        try:
            inverted = build_inverted_index(documents)
            print(f"Inverted index built ({len(inverted)} terms).")
        except Exception as e:
            print("Gagal membangun inverted index:", e)

    if build_incidence_matrix is not None:
        try:
            incidence, doc_ids = build_incidence_matrix(documents, vocabulary)
            print(f"Incidence matrix shape: {incidence.shape}")
        except Exception as e:
            print("Gagal membangun incidence matrix:", e)

    return documents, doc_id_map, vocabulary, inverted, incidence

def boolean_query_cli(inverted_index, all_doc_ids):
    """Loop interaktif untuk query Boolean dengan preprocessing (stemmer & stopwords)."""
    if inverted_index is None or boolean_retrieve is None:
        print("Boolean functionality tidak tersedia.")
        return

    stemmer, stop_words = (None, None)
    if initialize_preprocessing:
        try:
            stemmer, stop_words = initialize_preprocessing()
        except Exception as e:
            print("Peringatan: gagal inisialisasi preprocessing:", e)

    print("Masukkan query Boolean (misal: 'kriptografi AND keamanan', 'NOT proyek', 'sistem'). Ketik 'back' untuk kembali.")
    while True:
        q = input("Boolean query> ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit", "back"):
            break

        try:
            if stemmer and stop_words:
                res = boolean_retrieve(q, inverted_index, all_doc_ids, stemmer, stop_words)
            else:
                res = boolean_retrieve(q, inverted_index, all_doc_ids)
        except TypeError:
            res = boolean_retrieve(q, inverted_index, all_doc_ids)
        except Exception as e:
            print("Terjadi error saat menjalankan query:", e)
            continue

        print("Hasil:", res)

def vsm_query_cli(tfidf_matrix, idf_vector, term_to_idx, doc_ids, doc_map):
    """Simple loop to run VSM queries (query will be preprocessed inside this script)."""
    if calculate_tf_idf is None or rank_documents is None or query_to_tfidf_vector is None:
        print("VSM functionality tidak tersedia.")
        return
    stemmer, stop_words = (None, None)
    if initialize_preprocessing:
        try:
            stemmer, stop_words = initialize_preprocessing()
        except Exception:
            stemmer, stop_words = (None, None)

    print("Masukkan query untuk VSM. Ketik 'back' untuk kembali.")
    while True:
        q = input("VSM query> ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit", "back"):
            break

        if preprocess_query and stemmer is not None:
            try:
                tokens = preprocess_query(q, stemmer, stop_words)
            except Exception:
                tokens = q.lower().split()
        else:
            tokens = q.lower().split()

        q_text = " ".join(tokens)
        qvec = query_to_tfidf_vector(q_text, term_to_idx, idf_vector)
        ranking = rank_documents(qvec, tfidf_matrix, doc_ids)
        print(f"\nTop results for: '{q_text}'")
        for rank, (doc_id, score) in enumerate(ranking[:5], 1):
            print(f"{rank}. {doc_id} ({doc_map.get(doc_id,'-')})  score={score:.6f}")
        print("-" * 40)

def run_vsm_and_return():
    """Construct TF-IDF from processed docs and return matrix + mapping for queries."""
    ensure_dirs()
    res = build_indices_from_processed()
    if not res:
        print("Tidak dapat memuat dokumen terproses.")
        return None
    documents, doc_map, vocabulary, inverted, incidence = res

    if calculate_tf_idf is None:
        print("Fungsi calculate_tf_idf tidak tersedia pada vsm_ir.py")
        return None

    tfidf_matrix, idf_vector, term_to_idx, doc_ids = calculate_tf_idf(documents, vocabulary)
    print("TF-IDF matrix built:", tfidf_matrix.shape)
    return tfidf_matrix, idf_vector, term_to_idx, doc_ids, doc_map

def evaluate_sample_queries():
    """Jalankan evaluasi untuk beberapa skenario uji (contoh)."""
    vsm_res = run_vsm_and_return()
    if not vsm_res:
        print("VSM tidak tersedia; evaluasi dibatalkan.")
        return
    tfidf_matrix, idf_vector, term_to_idx, doc_ids, doc_map = vsm_res

    queries_gold = [
        ("manajemen proyek teknologi", ["D5"]),
        ("sistem terdistribusi", ["D3"]),
        ("algoritma enkripsi rsa", ["D4"]),
    ]

    for q, gold in queries_gold:
        qvec = query_to_tfidf_vector(q, term_to_idx, idf_vector)
        ranking = rank_documents(qvec, tfidf_matrix, doc_ids)
        retrieved = [d for d, s in ranking]
        print(f"\nQuery: {q}")
        print("Top 5:", ranking[:5])
        if eval_search:
            P, R, F1, nDCG = eval_search(retrieved, gold, k=5)
            print(f" Eval -> P:{P:.4f}, R:{R:.4f}, F1:{F1:.4f}, nDCG:{nDCG:.4f}")
        else:
            print("Eval module not available. Skipping metrics.")

def interactive_vsm_search_loop():
    vsm_res = run_vsm_and_return()
    if not vsm_res:
        return
    tfidf_matrix, idf_vector, term_to_idx, doc_ids, doc_map = vsm_res
    vsm_query_cli(tfidf_matrix, idf_vector, term_to_idx, doc_ids, doc_map)

def main_menu():
    ensure_dirs()
    while True:
        print("\n=== UTS STKI - MAIN MENU ===")
        print("1) Preprocess all documents (generate data_processed/CLEAN_*.txt)")
        print("2) Build indices (inverted index + incidence matrix) and show stats")
        print("3) Boolean query (interactive)")
        print("4) Build VSM (TF-IDF) and run example query")
        print("5) Interactive VSM search (top-K)")
        print("6) Run evaluation examples (Precision/Recall/F1/nDCG)")
        print("0) Exit")
        choice = input("Pilih nomor: ").strip()
        if choice == "1":
            run_preprocessing_and_save()
        elif choice == "2":
            build_indices_from_processed()
        elif choice == "3":
            res = build_indices_from_processed()
            if res:
                documents, doc_map, vocabulary, inverted, incidence = res
                all_doc_ids = sorted(documents.keys())
                boolean_query_cli(inverted, all_doc_ids)
        elif choice == "4":
            vsm_res = run_vsm_and_return()
            if vsm_res:
                tfidf_matrix, idf_vector, term_to_idx, doc_ids, doc_map = vsm_res
                q = input("Masukkan contoh query VSM (atau enter untuk 'sistem terdistribusi'): ").strip() or "sistem terdistribusi"
                qvec = query_to_tfidf_vector(q, term_to_idx, idf_vector)
                ranking = rank_documents(qvec, tfidf_matrix, doc_ids)
                print("Top 5 results:")
                for r, (doc_id, score) in enumerate(ranking[:5], 1):
                    print(f" {r}. {doc_id} ({doc_map.get(doc_id)}) score={score:.6f}")
        elif choice == "5":
            interactive_vsm_search_loop()
        elif choice == "6":
            evaluate_sample_queries()
        elif choice == "0":
            print("Keluar. Sampai jumpa.")
            break
        else:
            print("Pilihan tidak dikenali. Coba lagi.")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nDihentikan oleh pengguna. Keluar.")