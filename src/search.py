import os
import time
from preprocess import initialize_preprocessing, preprocess_query
from vsm_ir import load_processed_documents, calculate_tf_idf, query_to_tfidf_vector, rank_documents

PROCESSED_DIR = "data_processed"
K_TOP = 5  

def cli():
    print("SISTEM TEMU KEMBALI INFORMASI (STKI) - VSM Search [REAL MODEL]")
    print("---------------------------------------------------------------")

    start_time = time.time()
    
    stemmer, stop_words = initialize_preprocessing()

    docs, doc_map, vocabulary, raw_text_map = load_processed_documents(PROCESSED_DIR)
    if not docs:
        print("⚠️ Folder 'data_processed' kosong atau belum dibuat. Jalankan preprocess.py dulu.")
        return

    tfidf_matrix_doc, idf_vector, term_to_idx, doc_ids = calculate_tf_idf(docs, vocabulary)

    print(f"\n✅ Inisialisasi selesai ({len(docs)} dokumen, {len(vocabulary)} term).")
    print(f"Ukuran TF-IDF matrix: {tfidf_matrix_doc.shape}")
    print(f"Waktu inisialisasi: {time.time() - start_time:.2f} detik.")
    print("Ketik 'exit' untuk keluar.")
    print("---------------------------------------------------------------")

    while True:
        query_str = input("\nMasukkan query (ketik 'exit' untuk keluar): ").strip()

        if query_str.lower() == "exit":
            print("Terima kasih. Program diakhiri.")
            break

        if not query_str:
            print("⚠️ Query tidak boleh kosong.")
            continue

        processed_query_tokens = preprocess_query(query_str, stemmer, stop_words)
        if not processed_query_tokens:
            print("⚠️ Query hanya berisi stopword atau tidak valid. Coba kata lain.")
            continue

        print(f"Query terproses: {processed_query_tokens}")

        query_vector = query_to_tfidf_vector(" ".join(processed_query_tokens), term_to_idx, idf_vector)

        ranking = rank_documents(query_vector, tfidf_matrix_doc, doc_ids)

        print("\n--- Hasil Pencarian (Top 5) ---")
        print(f"{'Rank':<5}{'Doc ID':<8}{'Score':<10}{'Dokumen':<40}")
        print("-" * 80)

        found = False
        for rank, (doc_id, score) in enumerate(ranking[:K_TOP], 1):
            found = True
            doc_name = doc_map.get(doc_id, "Tidak diketahui")
            print(f"{rank:<5}{doc_id:<8}{score:.4f}   {doc_name}")

        if not found:
            print("Tidak ada dokumen relevan ditemukan.")
        print("-" * 80)

        from eval import evaluate_search_engine

    retrieved_docs = [doc_id for doc_id, score in ranking]
    gold_sets = {
        "kriptografi": ["D4"],
        "manajemen proyek": ["D5"],
        "sistem terdistribusi": ["D3"],
    }

    for key in gold_sets:
        if key in query_str.lower():
            evaluate_search_engine(retrieved_docs, gold_sets[key])
            break

if __name__ == "__main__":
    cli()
