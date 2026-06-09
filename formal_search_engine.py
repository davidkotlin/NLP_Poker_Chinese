import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import numpy as np
from rapidfuzz import fuzz
from core.reranker import RerankStrategy

# 關閉 sklearn 煩人的版本警告
# 可能遇到
# 1.Windows 環境下 KMeans 的記憶體洩漏警告
# 2.模型版本不一致警告 (Version Mismatch)
# 3.特徵名稱驗證警告 (Feature Names Missing)
warnings.filterwarnings("ignore", category=UserWarning)

# 依賴注入：引入工具箱
from core.text_processor import TextPreprocessor
from core.vectorizer import VectorizationStrategy
# from core.clustering import ClusteringStrategy
from core.embedding import EmbeddingStrategy, reciprocal_rank_fusion
class PokerSearchEngine:
    '''搜尋引擎大腦：負責指揮各個工具，執行兩階段檢索'''
    
    def __init__(self, preprocessor, vectorizer, embedder, reranker, model_dir="models"):
        print("啟動德州撲克搜尋大腦...")
        
        # 1. 接收從外部注入的武裝工具 (這些工具都已經自帶裝備/權重了)
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        # self.clusterer = clusterer
        self.embedder = embedder
        self.reranker = reranker
        # 2. 載入知識庫文本 (因為這是單純的表格資料，交由大腦親自管理)
        print("載入知識庫文本...")
        self.df = pd.read_pickle(f"{model_dir}/poker_data_clustered.pkl")
        print("系統上線，準備完畢！")

    # def search(self, query, top_k=3):
    #     '''舊版：純 TF-IDF 搜尋 (保留做對照)
    #     # =========================================================
    #     # 階段一：使用者輸入前處理 
    #     # =========================================================
    #     cleaned_query_list = self.preprocessor.clean([query])
    #     cleaned_query = cleaned_query_list[0]
    #     query_vec = self.vectorizer.transform([cleaned_query])
        
    #     # =========================================================
    #     # 階段二：KMeans 漏斗過濾 (五選二擴大檢索範圍！)
    #     # =========================================================
    #     # 呼叫我們新寫的方法，取得最接近的前 2 名群組 (例如回傳 [1, 3])
    #     target_clusters = self.clusterer.predict_top_k_clusters(query_vec, top_k=2)
        
    #     # 🎯 看這裡！使用 Pandas 的 .isin() 來過濾多個群組
    #     # 找出 DataFrame 中，Cluster_ID 包含在 target_clusters 裡面的所有 Index
    #     cluster_indices = self.df.index[self.df['Cluster_ID'].isin(target_clusters)].tolist()
        
    #     # 從 Vectorizer 肚子裡抽出這「兩群」的矩陣列
    #     filtered_matrix = self.vectorizer.tfidf_matrix[cluster_indices]
    #     filtered_df = self.df.iloc[cluster_indices].reset_index(drop=True)

    #     # =========================================================
    #     # 階段三：精確計算 Cosine Similarity 並排序回傳
    #     # =========================================================
    #     # (這裡跟你剛剛修改過的雙引擎或純 TF-IDF 寫法一模一樣，不用動！)
    #     cosine_scores = cosine_similarity(query_vec, filtered_matrix).flatten()
        
    #     # 取前 top_k 名
    #     top_indices = cosine_scores.argsort()[-top_k:][::-1]
        
    #     results = []
    #     for idx in top_indices:
    #         score = cosine_scores[idx]
    #         if score > 0.05: 
    #             results.append({
    #                 "clusters_searched": target_clusters, # 可以順便印出這次搜了哪兩群
    #                 "article_cluster": filtered_df.loc[idx, 'Cluster_ID'], # 這篇文章實際上屬於哪一群
    #                 "score": round(score, 4),
    #                 "content": filtered_df.loc[idx, '欄位 A (text)']
    #             })
    #     '''
    #     cleaned_query = self.preprocessor.clean([query])[0]
    #     q_tfidf = self.vectorizer.transform([cleaned_query])
    #     tfidf_scores = cosine_similarity(q_tfidf, self.vectorizer.tfidf_matrix).flatten()

    #     q_emb = self.embedder.transform([query])      # ⚠️ 原始 query，不要斷詞
    #     emb_scores = self.embedder.similarity(q_emb)

    #     final = reciprocal_rank_fusion([tfidf_scores, emb_scores])   # 全庫融合，不再過濾群組
    #     top_indices = final.argsort()[-top_k:][::-1]

    #     results = []
    #     for idx in top_indices:
    #         results.append({
    #             "score": round(float(final[idx]), 4),
    #             "content": self.df.iloc[idx]['欄位 A (text)'],
    #         })        
    #     return results
    # 3. search() 改成「粗篩 candidate_k 篇 → reranker 精排取 top_k」
    def search(self, query, top_k=3, candidate_k=20):
        cleaned = self.preprocessor.clean([query])[0]
        tfidf_scores = cosine_similarity(self.vectorizer.transform([cleaned]),
                                        self.vectorizer.tfidf_matrix).flatten()
        emb_scores = self.embedder.similarity(self.embedder.transform([query]))
        fused = reciprocal_rank_fusion([tfidf_scores, emb_scores])

        # 第一關：粗篩前 candidate_k 篇候選
        cand_idx = fused.argsort()[-candidate_k:][::-1]
        candidates = [(int(i), str(self.df.iloc[i]['欄位 A (text)'])) for i in cand_idx]

        # 第二關：reranker 精排，取前 top_k
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        return [{"score": round(s, 4),
                "content": text,
                "label": self.df.iloc[idx]['欄位 C (my_label) 給你自己對答案用的']}
                for idx, text, s in reranked]

# ==========================================
# 執行區塊 (模擬伺服器啟動)
# ==========================================
if __name__ == "__main__":
    model_dir = "models"
    
    try:
        # ---------------------------------------------------------
        # 1. 工具實例化與裝備載入 (裝配線)
        # ---------------------------------------------------------
        print("正在啟動各項核心模組...")
        
        # 文字清理器 (無權重，只需設定檔)
        my_preprocessor = TextPreprocessor(
            dict_path="dict/custom_words.txt", 
            stop_path="dict/stopwords.txt"
        )
        
        # 分群器 (喚醒模型)
        # my_clusterer = ClusteringStrategy()
        # my_clusterer.load_model(f"{model_dir}/kmeans_model.joblib")
        my_embedder = EmbeddingStrategy()
        my_embedder.load_model(f"{model_dir}/doc_embeddings.joblib")
        # 向量化器 (喚醒字典與矩陣)
        my_vectorizer = VectorizationStrategy()
        my_vectorizer.load_model(
            vec_path=f"{model_dir}/tfidf_vectorizer.joblib", 
            mat_path=f"{model_dir}/tfidf_matrix.joblib"
        )
        my_reranker = RerankStrategy()
        # ---------------------------------------------------------
        # 2. 核心大腦啟動
        # ---------------------------------------------------------
        # 把裝備好的神兵利器，全部交給大腦
        engine = PokerSearchEngine(my_preprocessor, my_vectorizer, my_embedder, my_reranker, model_dir)
        
        # ---------------------------------------------------------
        # 3. 測試玩家查詢
        # ---------------------------------------------------------
        # user_query = "對手在河牌重注，我該抓雞嗎？"
        # print(f"\n玩家詢問: {user_query}")
        # print("-" * 50)
        while True:
            user_query = input("\n🔍 請輸入你想查詢的撲克問題 (例如：怎麼算底池賠率？)：")
            if user_query.lower() in ['exit', 'q', 'quit']:
                print("系統關閉，下次見！👋")
                break
            if not user_query.strip():
                continue
        
            results = engine.search(user_query)
            
            if not results:
                print("抱歉，知識庫中找不到高度相關的文章。")
            else:
                # UX 優化：先把大腦這次「五選二」鎖定的群組印出來，讓玩家知道系統在幹嘛
                # searched_clusters = results[0]['clusters_searched']
                # print(f"\n💡 系統已鎖定第 {searched_clusters} 群組進行深度檢索，為您找到以下結果：")
                print("-" * 60)
                
                # 走訪結果清單
                for i, res in enumerate(results):
                    score = res['score']
                    # article_cluster = res['article_cluster']
                    
                    # 取出完整內容，不使用 [:150] 切片
                    full_content = str(res['content'])

                    # 使用全新的、乾淨的排版
                    # print(f"[{i+1}] 🎯 相似度: {score} | 📁 所屬群組: {article_cluster}")
                    print(f'[{i+1}] 🎯 相似度: {score}')
                    # print(f"📝 內容: {content_preview}...\n")
                    print(f"📝 內容:\n{full_content}")
                    print("-" * 60)
                
    except FileNotFoundError as e:
        print(f"\n系統啟動失敗：{e}")
        print("請確認您是否已經先執行過 formal_train_model.py 並且產生了 models 資料夾！")