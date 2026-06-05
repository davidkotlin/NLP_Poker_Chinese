'''語意向量化 + 混合檢索融合模組

放進 core/ 與 vectorizer.py 並列使用。
安裝依賴：pip install sentence-transformers rank_bm25

⚠️ 重要前處理差異：
- TF-IDF / BM25：餵「jieba 斷好 + 去停用詞」的文字
- Embedding   ：餵「原始自然句」(模型自己 tokenize，先斷詞反而破壞語意)
  → 所以 corpus 要保留「原始版」與「斷詞版」兩份
'''
import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


# ==========================================
# 模組：語意向量化 (Sentence Embedding / bi-encoder)
# ==========================================
class EmbeddingStrategy:
    '''負責語意向量化，補 TF-IDF「看不懂意圖」的洞。介面刻意對齊 VectorizationStrategy。'''

    def __init__(self, model_name="BAAI/bge-base-zh-v1.5"):
        # 中文可選：BAAI/bge-small-zh-v1.5(輕)、BAAI/bge-base-zh-v1.5(較準)、
        #           shibing624/text2vec-base-chinese
        print(f"載入語意模型 {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None  # 全庫向量存自己肚子裡

    def fit_transform(self, raw_corpus):
        '''【訓練用】把整個知識庫(原始句!)編碼成向量並記住
        normalize_embeddings=True → 之後算內積就等於 cosine 相似度'''
        print("計算全庫語意向量...")
        self.doc_embeddings = self.model.encode(
            list(raw_corpus), normalize_embeddings=True, show_progress_bar=True
        )
        return self.doc_embeddings

    def transform(self, raw_query_list):
        '''【搜尋用】把 query(原始句!)編碼成向量'''
        return self.model.encode(list(raw_query_list), normalize_embeddings=True)

    def similarity(self, query_vec):
        '''回傳 query 對全庫每一篇的語意相似度 (因為已正規化，內積=cosine)'''
        return (self.doc_embeddings @ query_vec[0]).flatten()

    def save_model(self, emb_path):
        '''模型本體不存(從 HuggingFace 載)，只存算好的全庫向量'''
        joblib.dump(self.doc_embeddings, emb_path)

    def load_model(self, emb_path):
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"找不到語意向量檔 {emb_path}，請先訓練！")
        self.doc_embeddings = joblib.load(emb_path)


# ==========================================
# 融合函式：把兩個引擎的分數合成一個排序
# ==========================================
def _minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


def weighted_fusion(tfidf_scores, emb_scores, alpha=0.5):
    '''做法 A：加權正規化(直接對應「兩個分數加權」)
    先各自 min-max 拉到 0~1 再加權，否則 embedding 的大分數會輾壓 TF-IDF。
    alpha 越大 → 越偏向術語精準比對；越小 → 越偏向語意。建議從 0.4~0.5 開始調。'''
    return alpha * _minmax(tfidf_scores) + (1 - alpha) * _minmax(emb_scores)


def reciprocal_rank_fusion(score_lists, k=60):
    '''做法 B：RRF(推薦預設) —— 只看名次不看分數大小，天生免疫尺度問題
    score_lists：[tfidf_scores, emb_scores, ...] 每個都是「對全庫每篇的相似度」陣列
    最終分數 = Σ 1/(k + 該篇在各引擎的名次)'''
    fused = np.zeros(len(score_lists[0]))
    for s in score_lists:
        s = np.asarray(s, dtype=float)
        order = np.argsort(s)[::-1]            # 由大到小的索引
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))   # 名次：0 = 最相關
        fused += 1.0 / (k + ranks + 1)
    return fused


# ==========================================
# 接進 search() 的範例 (示意，非可獨立執行)
# ==========================================
def hybrid_search_example(query, preprocessor, vectorizer, embedder, df,
                          top_k=3, mode="rrf", alpha=0.5):
    from sklearn.metrics.pairwise import cosine_similarity

    # --- 引擎一：TF-IDF (吃斷詞版) ---
    cleaned = preprocessor.clean([query])           # jieba + 停用詞
    q_tfidf = vectorizer.transform(cleaned)
    tfidf_scores = cosine_similarity(q_tfidf, vectorizer.tfidf_matrix).flatten()

    # --- 引擎二：Embedding (吃原始句!) ---
    q_emb = embedder.transform([query])             # 注意：原始 query，不要斷詞
    emb_scores = embedder.similarity(q_emb)

    # --- 融合 ---
    if mode == "rrf":
        final = reciprocal_rank_fusion([tfidf_scores, emb_scores])
    else:
        final = weighted_fusion(tfidf_scores, emb_scores, alpha=alpha)

    top_idx = final.argsort()[-top_k:][::-1]
    return [
        {
            "score": round(float(final[i]), 4),
            "tfidf": round(float(tfidf_scores[i]), 4),
            "emb": round(float(emb_scores[i]), 4),
            "content": df.iloc[i]['欄位 A (text)'],
        }
        for i in top_idx
    ]