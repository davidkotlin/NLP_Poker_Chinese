'''Re-ranker：用 cross-encoder 對候選文章做第二關精排

跟 embedding/tfidf 不同：bi-encoder(embedding)是把 query 和文章「各自」壓成向量再比，
快但判斷力弱；cross-encoder 是把 query 和每篇文章「黏在一起」一起讀，判斷力強但慢，
所以只用在第一關粗篩出來的少量候選(例如前 20 篇)上。

安裝：pip install sentence-transformers  (你裝 embedding 時已經有了)
模型 BAAI/bge-reranker-base 第一次跑會自動下載(約 280MB)，純 CPU 可跑，
重排 ~20 篇大約 1~3 秒。想更快可換更小的、想更準可換 bge-reranker-v2-m3(較大)。
'''
import numpy as np
from sentence_transformers import CrossEncoder


class RerankStrategy:
    '''第二關精排：把粗篩出的候選逐一和 query 一起精讀，重新打分排序'''

    def __init__(self, model_name="BAAI/bge-reranker-base"):
        print(f"載入 Re-ranker 模型 {model_name} ...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_k=3):
        '''query：使用者原始問句(不要斷詞)
        candidates：[(index, text), ...] 第一關粗篩出來的候選
        回傳依精排分數由高到低的前 top_k 個 (index, text, score)
        score 已用 sigmoid 壓成 0~1，越高越相關'''
        if not candidates:
            return []
        pairs = [(query, text) for _, text in candidates]
        logits = np.asarray(self.model.predict(pairs), dtype=float)
        scores = 1.0 / (1.0 + np.exp(-logits))          # logit -> 0~1 方便當「信心」顯示
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(idx, text, float(s)) for (idx, text), s in ranked[:top_k]]
