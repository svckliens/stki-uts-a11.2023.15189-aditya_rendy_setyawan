import os
from collections import Counter
import numpy as np
from preprocess import initialize_preprocessing, preprocess_query

PROCESSED_DIR = 'data_processed'

def load_processed_documents(directory):
    documents = {}
    doc_id_map = {}
    all_terms = set()
    
    file_list = sorted([f for f in os.listdir(directory) if f.startswith('CLEAN_') and f.endswith('.txt')])
    
    for i, filename in enumerate(file_list):
        doc_id = f'D{i+1}'
        filepath = os.path.join(directory, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tokens = f.read().split()
            documents[doc_id] = tokens
            doc_id_map[doc_id] = filename.replace('CLEAN_', '').replace('.txt', '')
            all_terms.update(tokens)
        except Exception as e:
            print(f"Gagal memuat {filename}: {e}")
            
    vocabulary = sorted(list(all_terms))
    return documents, doc_id_map, vocabulary

def build_inverted_index(documents):
    inverted_index = {}
    for doc_id, tokens in documents.items():
        for term in set(tokens):
            inverted_index.setdefault(term, []).append(doc_id)
    for term in inverted_index:
        inverted_index[term].sort()
    return inverted_index

def build_incidence_matrix(documents, vocabulary):
    doc_ids = sorted(documents.keys())
    matrix = np.zeros((len(vocabulary), len(doc_ids)), dtype=int)
    term_to_idx = {term: i for i, term in enumerate(vocabulary)}
    doc_to_idx = {doc_id: j for j, doc_id in enumerate(doc_ids)}
    
    for doc_id, tokens in documents.items():
        doc_idx = doc_to_idx[doc_id]
        for term in set(tokens):
            if term in term_to_idx:
                matrix[term_to_idx[term], doc_idx] = 1
                
    return matrix, doc_ids

def intersect(a, b): return sorted(list(set(a) & set(b)))
def union(a, b): return sorted(list(set(a) | set(b)))
def complement(a, all_docs): return sorted(list(set(all_docs) - set(a)))

def boolean_retrieve(query_str, inverted_index, all_doc_ids, stemmer, stop_words):
    processed_query = preprocess_query(query_str, stemmer, stop_words)
    if not processed_query:
        return []

    if len(processed_query) == 2 and processed_query[0].upper() == 'NOT':
        term = processed_query[1]
        postings = inverted_index.get(term, [])
        return complement(postings, all_doc_ids)
    
    if len(processed_query) == 3:
        term1, operator, term2 = processed_query[0], processed_query[1].upper(), processed_query[2]
        postings1 = inverted_index.get(term1, [])
        postings2 = inverted_index.get(term2, [])
        if operator == 'AND':
            return intersect(postings1, postings2)
        elif operator == 'OR':
            return union(postings1, postings2)

    if len(processed_query) == 1:
        return inverted_index.get(processed_query[0], [])
    
    print(f"Peringatan: Query '{query_str}' tidak dikenali.")
    return []

def calculate_precision_recall(retrieved, relevant):
    retrieved, relevant = set(retrieved), set(relevant)
    tp = len(retrieved & relevant)
    fp = len(retrieved - relevant)
    fn = len(relevant - retrieved)
    precision = tp / (len(retrieved) or 1)
    recall = tp / (len(relevant) or 1)
    f1 = 2 * precision * recall / ((precision + recall) or 1)
    return precision, recall, f1, tp, fp, fn

if __name__ == "__main__":
    stemmer, stop_words = initialize_preprocessing()
    
    docs, doc_map, vocabulary = load_processed_documents(PROCESSED_DIR)
    all_doc_ids = sorted(docs.keys())
    
    if not docs:
        print("Folder 'data_processed' kosong. Jalankan preprocess.py dulu!")
        exit()
        
    inverted_index = build_inverted_index(docs)
    matrix, doc_ids = build_incidence_matrix(docs, vocabulary)
    
    print("="*90)
    print("DEMO SOAL 03 — BOOLEAN RETRIEVAL, INVERTED INDEX, & EVALUASI")
    print("="*90)
    print(f"Jumlah Term: {len(vocabulary)} | Jumlah Dokumen: {len(all_doc_ids)}")
    print(f"Dokumen: {list(doc_map.values())}\n")
    
    queries = [
        ("informasi AND proyek", ['D5']),
        ("kriptografi OR dekripsi", ['D4']),
        ("NOT proyek", [d for d in all_doc_ids if d != 'D5'])
    ]
    
    print(f"{'Query':35} | {'Precision':9} | {'Recall':7} | {'F1':5} | Hasil Dokumen")
    print("-"*90)
    
    for query_raw, gold_set in queries:
        result = boolean_retrieve(query_raw, inverted_index, all_doc_ids, stemmer, stop_words)
        P, R, F1, TP, FP, FN = calculate_precision_recall(result, gold_set)
        print(f"{query_raw:<35} | {P:9.3f} | {R:7.3f} | {F1:5.3f} | {result}")

    print("="*90)
    print("Semua query sudah otomatis di-stem agar konsisten dengan hasil preprocessing.")