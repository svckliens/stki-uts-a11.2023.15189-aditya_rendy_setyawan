import os
import re
import glob
from collections import Counter
from typing import List, Set, Any, Dict, Tuple
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def get_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

def get_stop_words() -> Set[str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, '..', 'data', 'stopwords.txt')

    stop_words = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    stop_words.add(word)
    except FileNotFoundError:
        print(f"[PERINGATAN] stopwords.txt tidak ditemukan di {filepath}. Menggunakan set kosong.")
    return stop_words

def list_all_documents() -> List[str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    document_paths = glob.glob(os.path.join(data_dir, '*.txt'))

    document_paths = [
        p for p in document_paths
        if not os.path.basename(p).lower() == 'stopwords.txt'
    ]
    
    if not document_paths:
        print(f"[PERINGATAN] Tidak ada dokumen ditemukan di {data_dir}")
    else:
        print(f"[INFO] Ditemukan {len(document_paths)} dokumen di {data_dir}")
    
    return document_paths


def load_all_documents(document_paths: List[str]) -> Dict[str, str]:
    documents = {}
    for path in document_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            doc_id = os.path.basename(path)
            documents[doc_id] = content
        except Exception as e:
            print(f"ERROR membaca {path}: {e}")
    return documents

def tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

def remove_stop_words(tokens: List[str], stop_words: Set[str]) -> List[str]:
    return [t for t in tokens if t not in stop_words]

def stem_text(tokens: List[str], stemmer: Any) -> List[str]:
    return [stemmer.stem(t) for t in tokens]

def preprocess_document(text: str, stemmer: Any, stop_words: Set[str]) -> List[str]:
    tokens = tokenize_text(text)
    tokens = remove_stop_words(tokens, stop_words)
    tokens = stem_text(tokens, stemmer)
    return tokens

def preprocess_query(text: str, stemmer: Any, stop_words: Set[str]) -> List[str]:
    if not text or not text.strip():
        return []
    tokens = tokenize_text(text)
    tokens = remove_stop_words(tokens, stop_words)
    tokens = stem_text(tokens, stemmer)
    return tokens

def initialize_preprocessing() -> Tuple[Any, Set[str]]:
    stemmer = get_stemmer()
    stop_words = get_stop_words()
    return stemmer, stop_words

def get_processed_corpus() -> Dict[str, List[str]]:
    stemmer, stop_words = initialize_preprocessing()
    
    doc_paths = list_all_documents()
    
    raw_documents = load_all_documents(doc_paths)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(current_dir, '..', 'data_processed')
    os.makedirs(processed_dir, exist_ok=True)

    processed_corpus = {}
    print(f"Memproses {len(raw_documents)} dokumen...")

    for doc_id, text in raw_documents.items():
        processed_tokens = preprocess_document(text, stemmer, stop_words)
        processed_corpus[doc_id] = processed_tokens

        clean_filename = f"CLEAN_{os.path.splitext(doc_id)[0]}.txt"
        clean_path = os.path.join(processed_dir, clean_filename)
        try:
            with open(clean_path, 'w', encoding='utf-8') as f:
                f.write(' '.join(processed_tokens))
            print(f"[OK] {clean_filename} disimpan ({len(processed_tokens)} token).")
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan {clean_filename}: {e}")
    
    print(f"\nSemua dokumen tersimpan di folder: {processed_dir}")
    return processed_corpus

if __name__ == '__main__':
    get_processed_corpus()

    print("--- TEST PREPROCESS ---")
    stemmer, stop_words = initialize_preprocessing()
    print(f"Stopword dimuat: {len(stop_words)} kata.")
    print("Stemmer siap digunakan.")

    doc_paths = list_all_documents()
    print(f"Ditemukan {len(doc_paths)} dokumen txt.")