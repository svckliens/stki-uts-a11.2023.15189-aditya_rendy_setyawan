import os
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED_DIR = 'data_processed'
K_TOP = 5 

def load_processed_documents(directory):
    documents = {}
    doc_id_map = {}
    raw_text_map = {}
    all_terms = set()
    
    file_list = sorted([f for f in os.listdir(directory) if f.startswith('CLEAN_') and f.endswith('.txt')])
    
    for i, filename in enumerate(file_list):
        doc_id = f'D{i+1}'
        filepath = os.path.join(directory, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                tokens = content.split() 
            
            documents[doc_id] = tokens
            doc_id_map[doc_id] = filename.replace('CLEAN_', '').replace('.txt', '')
            raw_text_map[doc_id] = content[:120].replace('\n', ' ') + '...'
            all_terms.update(tokens)
        except Exception as e:
            print(f"Gagal memuat {filename}: {e}")
            
    vocabulary = sorted(list(all_terms))
    return documents, doc_id_map, vocabulary, raw_text_map

def calculate_tf_idf(documents, vocabulary):
    N = len(documents)
    doc_ids = sorted(documents.keys())

    tf_matrix_data, tf_matrix_rows, tf_matrix_cols = [], [], []
    df = Counter()

    term_to_idx = {term: i for i, term in enumerate(vocabulary)}
    
    for j, doc_id in enumerate(doc_ids):
        tokens = documents[doc_id]
        term_counts = Counter(tokens)
    
        for term in set(tokens):
            df[term] += 1
            
        for term, count in term_counts.items():
            if term in term_to_idx:
                tf = 1 + np.log10(count) if count > 0 else 0

                term_idx = term_to_idx[term]
                
                tf_matrix_data.append(tf)
                tf_matrix_rows.append(term_idx)
                tf_matrix_cols.append(j)       

    idf_vector = np.zeros(len(vocabulary))
    for term, idx in term_to_idx.items():
        
        idf_vector[idx] = np.log10(N / (df[term] or 1))        
    
    tf_matrix = csr_matrix((tf_matrix_data, (tf_matrix_rows, tf_matrix_cols)), 
                           shape=(len(vocabulary), N))

    tfidf_matrix = tf_matrix.multiply(idf_vector[:, np.newaxis])
    
    return tfidf_matrix, idf_vector, term_to_idx, doc_ids

def query_to_tfidf_vector(query_str, term_to_idx, idf_vector):
    query_tokens = query_str.lower().split()
    query_counts = Counter(query_tokens)
    
    query_vector = np.zeros(len(term_to_idx))
    
    for term, count in query_counts.items():
        if term in term_to_idx:
            term_idx = term_to_idx[term]
            
            tf_q = 1 + np.log10(count) if count > 0 else 0
            
            tfidf_q = tf_q * idf_vector[term_idx]
            
            query_vector[term_idx] = tfidf_q

    return csr_matrix(query_vector).transpose()

def rank_documents(query_vector, tfidf_matrix_doc, doc_ids):
    similarities = cosine_similarity(query_vector.transpose(), tfidf_matrix_doc.transpose())[0]
    
    ranking = sorted(zip(doc_ids, similarities), key=lambda x: x[1], reverse=True)
    
    return ranking

def calculate_map_and_precision_at_k(ranking, gold_set, k):
    retrieved_at_k = [doc_id for doc_id, score in ranking[:k]]
    relevant_set = set(gold_set)
    
    tp_at_k = len(set(retrieved_at_k).intersection(relevant_set))
    precision_at_k = tp_at_k / k
    
    ap = 0.0
    relevant_count = 0
    for i, doc_id in enumerate(retrieved_at_k):
        if doc_id in relevant_set:
            relevant_count += 1
            ap += relevant_count / (i + 1)
            
    ap = ap / (len(relevant_set) or 1) 

    return precision_at_k, ap

if __name__ == "__main__":
    
    docs, doc_map, vocabulary, raw_text_map = load_processed_documents(PROCESSED_DIR)
    all_doc_ids = sorted(docs.keys())
    
    if not docs:
        print("Pastikan folder 'data_processed' ada dan berisi file CLEAN_*.txt.")
        exit()
        
    tfidf_matrix_doc, idf_vector, term_to_idx, matrix_doc_ids = calculate_tf_idf(docs, vocabulary)
    
    queries_to_test = [
        ("manajemen proyek teknologi", ['D5', 'D1']),
        ("sistem terdistribusi", ['D3']),
        ("algoritma enkripsi rsa", ['D4']),
        ("sistem informasi", ['D2'])
    ]
    
    print("="*80)
    print("SOAL 04: VECTOR SPACE MODEL (VSM) & COSINE SIMILARITY")
    print("="*80)
    print(f"Total Dokumen (N): {len(all_doc_ids)}")
    print(f"Ukuran Vocabulary (V): {len(vocabulary)}")
    print(f"Ukuran Matriks TF-IDF (V x N): {tfidf_matrix_doc.shape[0]} x {tfidf_matrix_doc.shape[1]}")
    print("-" * 80)
    
    total_map = 0
    
    for i, (query_str, gold_set) in enumerate(queries_to_test):
        
        query_vector = query_to_tfidf_vector(query_str, term_to_idx, idf_vector)
        
        ranking = rank_documents(query_vector, tfidf_matrix_doc, matrix_doc_ids)
        
        print(f"\nQUERY {i+1}: '{query_str.upper()}'")
        print(f"  Gold Relevant Set: {gold_set}")
        print("-" * 50)
        print(f"{'Rank':<5}{'Doc ID':<8}{'Cosine Sim':<15}{'Snippet (120 char)':<50}")
        print("-" * 80)
        
        top_k_results = ranking[:K_TOP]
        retrieved_docs_k = [doc_id for doc_id, score in top_k_results]
        
        for rank, (doc_id, score) in enumerate(top_k_results):
            doc_name_snippet = raw_text_map.get(doc_id, "N/A")
            
            mark = '*' if doc_id in gold_set else ' '
            
            print(f"{mark}{rank+1:<4}{doc_id:<8}{score:.6f}{'':<4}{doc_name_snippet}")
            
        print("-" * 80)
        
        P_at_k, AP = calculate_map_and_precision_at_k(ranking, gold_set, K_TOP)
        
        print(f"METRIK EVALUASI (K={K_TOP}):")
        print(f"  > Precision@{K_TOP}: {P_at_k:.4f}")
        print(f"  > Average Precision (AP): {AP:.4f}")
        total_map += AP
        
    MAP = total_map / len(queries_to_test)
    print("\n" + "="*80)
    print(f"OVERALL METRIC:")
    print(f"  > Mean Average Precision (MAP) untuk {len(queries_to_test)} queries: {MAP:.4f}")
    print("="*80)