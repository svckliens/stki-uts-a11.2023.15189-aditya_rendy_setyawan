import numpy as np

def precision(retrieved_docs, relevant_docs):
    if not retrieved_docs:
        return 0.0
    return len(set(retrieved_docs) & set(relevant_docs)) / len(retrieved_docs)

def recall(retrieved_docs, relevant_docs):
    if not relevant_docs:
        return 0.0
    return len(set(retrieved_docs) & set(relevant_docs)) / len(relevant_docs)

def f1_score(P, R):
    return 0.0 if (P + R) == 0 else (2 * P * R) / (P + R)

def dcg_at_k(scores, k):
    scores = np.asarray(scores)[:k]
    if scores.size == 0:
        return 0.0
    return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))

def ndcg_at_k(ranking_scores, ideal_ranking_scores, k):
    dcg = dcg_at_k(ranking_scores, k)
    ideal_scores = sorted(ideal_ranking_scores, reverse=True)
    idcg = dcg_at_k(ideal_scores, k)
    return 0.0 if idcg == 0.0 else dcg / idcg

def evaluate_search_engine(retrieved_docs, gold_set, k=5):
    P = precision(retrieved_docs[:k], gold_set)
    R = recall(retrieved_docs[:k], gold_set)
    F1 = f1_score(P, R)

    relevance_scores = [1 if d in gold_set else 0 for d in retrieved_docs[:k]]
    ideal_scores = [1] * min(k, len(gold_set)) + [0] * max(0, k - len(gold_set))
    nDCG = ndcg_at_k(relevance_scores, ideal_scores, k)

    print(f"\n📊 EVALUASI HASIL PENCARIAN (Top-{k})")
    print(f"Precision@{k}: {P:.4f}")
    print(f"Recall@{k}:    {R:.4f}")
    print(f"F1-Score:      {F1:.4f}")
    print(f"nDCG@{k}:      {nDCG:.4f}")

    return P, R, F1, nDCG

if __name__ == "__main__":
    retrieved = ['D1', 'D5', 'D3', 'D2', 'D4']
    gold = ['D4', 'D3']
    evaluate_search_engine(retrieved, gold, k=5)
