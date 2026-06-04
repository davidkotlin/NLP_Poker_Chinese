import pandas as pd
import os
#引用文字處理模組
from core.text_processor import TextPreprocessor
#引用kmeans分群模組未來用
# from core.clustering import ClusteringStrategy
#引用精準術語向量化模組
from core.vectorizer import VectorizationStrategy
#引用語意向量化模組
from core.embedding import EmbeddingStrategy
# ==========================================
# 模組 1：專職處理文字清理 (包含字典與停用詞)
# ==========================================
# class TextPreprocessor:
#     '''這個類別專門負責文字清理與jieba斷詞，讓主控台不用管細節'''
#     def __init__(self, dict_path="dict/custom_words.txt", stop_path="dict/stopwords.txt"):
#         '''初始化文字處理器，載入自訂字典與停用詞表'''
#         print("初始化文字處理器...")
#         # 載入自訂字典
#         if os.path.exists(dict_path):
#             jieba.load_userdict(dict_path)
#         else:
#             print(f"警告：找不到自訂字典 {dict_path}")
            
#         # 載入停用詞
#         self.stopwords = self._load_stopwords(stop_path)

#     def _load_stopwords(self, filepath):
#         if not os.path.exists(filepath):
#             print(f"警告：找不到停用詞表 {filepath}")
#             return set()
#         with open(filepath, 'r', encoding='utf-8') as f:
#             return set([line.strip() for line in f if line.strip()])

#     def clean(self, corpus):
#         '''對輸入的文字列表進行清理：斷詞 + 過濾停用詞 + 去除空白'''
#         print("執行 Jieba 斷詞與停用詞過濾...")
#         cleaned_corpus = []
#         for text in corpus:
#             # 斷詞
#             words = jieba.lcut(str(text).lower())
#             # 過濾停用詞與空白
#             filtered_words = [w for w in words if w not in self.stopwords and w.strip()]
#             cleaned_corpus.append(" ".join(filtered_words))
#         return cleaned_corpus

# ==========================================
# 模組 2：專職處理向量化 (TF-IDF)
# ==========================================
# class VectorizationStrategy:
#     '''這個類別專門負責 TF-IDF 向量化，讓主控台不用管細節'''
#     def __init__(self):
#         self.vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
        
#     def fit_transform(self, cleaned_corpus):
#         '''這fit_transform方法與vectorizer.fit_transform不同
#         用它來封裝vectorizer.fit_transform
#         讓PokerModelTrainer的主訓練區train可以多形'''
#         print("計算 TF-IDF 矩陣...")
#         return self.vectorizer.fit_transform(cleaned_corpus)

# ==========================================
# 模組 3：專職處理分群 (KMeans)
# ==========================================
# class ClusteringStrategy:
#     '''這個類別專門負責 KMeans 分群，讓主控台不用管細節'''
#     def __init__(self, num_clusters=5):
#         self.kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        
#     def fit_predict(self, matrix):
#         '''這fit_predict方法與kmeans.fit_predict不同
#         用它來封裝kmeans.fit_predict
#         讓PokerModelTrainer的主訓練區train可以多形
#         ==========================================
#         fit()完後還要再labels_取得分群標籤
#         fit_predict()會同時做完fit()和labels_，直接回傳分群標籤
#         '''
#         print(f"執行 KMeans 分群 (K={self.kmeans.n_clusters})...")
#         return self.kmeans.fit_predict(matrix)

# ==========================================
# 主控台：將三個模組組合起來 (Orchestrator)
# ==========================================
class PokerModelTrainer:
    '''主要訓練區'''
    def __init__(self, preprocessor, vectorizer, embedder):
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.embedder = embedder

        self.df = None

    def train(self, excel_path="data/撲克資料.xlsx", text_column="欄位 A (text)"):
        print(f"讀取資料庫: {excel_path}...")
        self.df = pd.read_excel(excel_path)
        corpus = self.df[text_column].tolist()

        # 1. 清理
        cleaned_corpus = self.preprocessor.clean(corpus)
        
        # 2. 專有名詞向量化 (注意：不需要用變數接矩陣了，因為它存在 vectorizer 自己肚子裡)
        tfidf_matrix = self.vectorizer.fit_transform(cleaned_corpus)
        
        # 3. 分群(未來)
        # self.df['Cluster_ID'] = self.clusterer.fit_predict(tfidf_matrix)
        #3.語意向量化 (注意： embedding 模型要餵原始自然句（它有自己的 tokenizer，你先斷詞反而會破壞語意）)
        self.embedder.fit_transform(corpus) 
        print("訓練管線執行完畢！")

    def save_models(self, save_dir="models"):
        '''儲存模型到指定目錄'''
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 🎯 看這裡！主控台只要下指令，工具自己會去存檔！
        self.vectorizer.save_model(
            vec_path=f"{save_dir}/tfidf_vectorizer.joblib", 
            mat_path=f"{save_dir}/tfidf_matrix.joblib"
        )
        # self.clusterer.save_model(f"{save_dir}/kmeans_model.joblib")
        self.embedder.save_model(f"{save_dir}/doc_embeddings.joblib")
        # DataFrame 是一般的資料表格，主控台負責直接存起來
        self.df.to_pickle(f"{save_dir}/poker_data_clustered.pkl")
        print(f"資料與模型已安全儲存至 {save_dir}/")

# ==========================================
# 執行區塊
# ==========================================
if __name__ == "__main__":
    # 就像組裝積木一樣，把需要的工具實例化
    my_preprocessor = TextPreprocessor(dict_path="dict/custom_words.txt", stop_path="dict/stopwords.txt")
    my_vectorizer = VectorizationStrategy()
    # my_clusterer = ClusteringStrategy(num_clusters=5)
    my_embedder = EmbeddingStrategy()
    # 把工具交給訓練員
    trainer = PokerModelTrainer(my_preprocessor, my_vectorizer, my_embedder)
    
    # 開始訓練並存檔
    trainer.train(excel_path="data/撲克資料.xlsx", text_column="欄位 A (text)")
    trainer.save_models()