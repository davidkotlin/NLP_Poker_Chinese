'''評測腳本 v2：對「文章句子」算 recall@3 (不靠 label)

每題在題庫填一段正解文章裡的獨特句子(expected_snippet，可用 | 放多個)，
評測就看前三名結果的「內文」有沒有出現它(用 rapidfuzz 模糊比對，容許小改動)。
這樣 label 寫得粗、或不確定對不對，都不影響評測。

前置：models/ 已有訓練好的檔；撲克評測題庫_v2.xlsx 放專案根目錄。
執行：python eval_recall_v2.py
'''
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

from core.text_processor import TextPreprocessor
from core.vectorizer import VectorizationStrategy
from core.embedding import EmbeddingStrategy, reciprocal_rank_fusion
from core.reranker import RerankStrategy

MODEL_DIR = "models"
TEXT_COL = "欄位 A (text)"
LABEL_COL = "欄位 C (my_label) 給你自己對答案用的"
EVAL_FILE = "data/撲克評測題庫.xlsx"
TOP_K = 3
FUZZ_THRESHOLD = 85   # 模糊比對門檻，0~100，越高越嚴格

pre = TextPreprocessor("dict/custom_words.txt", "dict/stopwords.txt")
vec = VectorizationStrategy()
vec.load_model(f"{MODEL_DIR}/tfidf_vectorizer.joblib", f"{MODEL_DIR}/tfidf_matrix.joblib")
emb = EmbeddingStrategy()
emb.load_model(f"{MODEL_DIR}/doc_embeddings.joblib")
df = pd.read_pickle(f"{MODEL_DIR}/poker_data_clustered.pkl")
reranker = RerankStrategy()

# def search_top(query, k=TOP_K):
#     '''回傳前 k 名的 (內文, label, 語意信心)'''
#     cleaned = pre.clean([query])[0]
#     tfidf_scores = cosine_similarity(vec.transform([cleaned]), vec.tfidf_matrix).flatten()
#     emb_scores = emb.similarity(emb.transform([query]))
#     final = reciprocal_rank_fusion([tfidf_scores, emb_scores])
#     idx = final.argsort()[-k:][::-1]
#     return [(str(df.iloc[i][TEXT_COL]), df.iloc[i][LABEL_COL], round(float(emb_scores[i]), 3))
#             for i in idx]
def search_top(query, k=TOP_K, candidate_k=20):
    '''回傳前 k 名的 (內文, label, 語意信心)'''
    cleaned = pre.clean([query])[0]
    tfidf_scores = cosine_similarity(vec.transform([cleaned]), vec.tfidf_matrix).flatten()
    emb_scores = emb.similarity(emb.transform([query]))
    fused = reciprocal_rank_fusion([tfidf_scores, emb_scores])
    cand_idx = fused.argsort()[-candidate_k:][::-1]
    candidates = [(int(i), str(df.iloc[i][TEXT_COL])) for i in cand_idx]
    reranked = reranker.rerank(query, candidates, k)
    return [(text, df.iloc[idx][LABEL_COL], round(s, 3)) for idx, text, s in reranked]

def hit_check(expected_field, contents):
    '''任一 snippet 在任一篇前三名內文裡(模糊)出現，就算命中'''
    snippets = [s.strip() for s in str(expected_field).split("|") if s.strip()]
    for snip in snippets:
        for c in contents:
            if fuzz.partial_ratio(snip, c) >= FUZZ_THRESHOLD:
                return True, snip
    return False, None


def run():
    evalset = pd.read_excel(EVAL_FILE)
    hit = 0
    misses = []
    for _, r in evalset.iterrows():
        got = search_top(r["query"])
        contents = [g[0] for g in got]
        ok, matched = hit_check(r["expected_snippet"], contents)
        hit += ok
        print(("✅" if ok else "❌"), r["query"], f"(命中片語：{matched})" if ok else "")
        for content, lab, conf in got:
            print(f"      - [{lab}] (信心 {conf})  {content[:]}")
        if not ok:
            misses.append(r["query"])

    n = len(evalset)
    print("\n" + "=" * 52)
    print(f"recall@{TOP_K} = {hit}/{n} = {hit / n * 100:.1f}%   (對文章層級)")
    if misses:
        print("\n沒中的題目：")
        for q in misses:
            print("  -", q)


if __name__ == "__main__":
    run()