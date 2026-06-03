import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import jieba
import os

# ==========================================
# 模組 1：專職處理文字清理 (包含字典與停用詞)
# ==========================================
class TextPreprocessor:
    '''這個類別專門負責文字清理與jieba斷詞，讓主控台不用管細節'''
    def __init__(self, dict_path="dict/custom_words.txt", stop_path="dict/stopwords.txt"):
        '''初始化文字處理器，載入自訂字典與停用詞表'''
        print("初始化文字處理器...")
        # 載入自訂字典
        if os.path.exists(dict_path):
            jieba.load_userdict(dict_path)
        else:
            print(f"警告：找不到自訂字典 {dict_path}")
            
        # 載入停用詞
        self.stopwords = self._load_stopwords(stop_path)

    def _load_stopwords(self, filepath):
        if not os.path.exists(filepath):
            print(f"警告：找不到停用詞表 {filepath}")
            return set()
        with open(filepath, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f if line.strip()])

    def clean(self, corpus):
        '''對輸入的文字列表進行清理：斷詞 + 過濾停用詞 + 去除空白'''
        print("執行 Jieba 斷詞與停用詞過濾...")
        cleaned_corpus = []
        for text in corpus:
            # 斷詞
            words = jieba.lcut(str(text).lower())
            # 過濾停用詞與空白
            filtered_words = [w for w in words if w not in self.stopwords and w.strip()]
            cleaned_corpus.append(" ".join(filtered_words))
        return cleaned_corpus

# ==========================================
# 模組 2：專職處理向量化 (TF-IDF)
# ==========================================
class VectorizationStrategy:
    '''這個類別專門負責 TF-IDF 向量化，讓主控台不用管細節'''
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
        
    def fit_transform(self, cleaned_corpus):
        print("計算 TF-IDF 矩陣...")
        return self.vectorizer.fit_transform(cleaned_corpus)

# ==========================================
# 模組 3：專職處理分群 (KMeans)
# ==========================================
class ClusteringStrategy:
    '''這個類別專門負責 KMeans 分群，讓主控台不用管細節'''
    def __init__(self, num_clusters=5):
        self.kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        
    def fit_predict(self, matrix):
        '''這fit_predict方法與kmeans.fit_predict不同
        用它來封裝kmeans.fit_predict
        ==========================================
        fit()完後還要再labels_取得分群標籤
        fit_predict()會同時做完fit()和labels_，直接回傳分群標籤
        '''
        print(f"執行 KMeans 分群 (K={self.kmeans.n_clusters})...")
        return self.kmeans.fit_predict(matrix)

# ==========================================
# 主控台：將三個模組組合起來 (Orchestrator)
# ==========================================
class PokerModelTrainer:
    '''主要訓練區'''
    def __init__(self, preprocessor, vectorizer, clusterer):
        # 依賴注入 (Dependency Injection)：工具是從外面傳進來的，主控台不負責製造工具
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.clusterer = clusterer
        self.df = None
        self.tfidf_matrix = None

    def train(self, excel_path="data/撲克資料.xlsx", text_column="欄位 A (text)"):
        '''訓練流程：讀取資料 -> 清理 -> 向量化 -> 分群'''
        print(f"讀取資料庫: {excel_path}...")
        self.df = pd.read_excel(excel_path)
        corpus = self.df[text_column].tolist()

        # 1. 清理
        cleaned_corpus = self.preprocessor.clean(corpus)
        # 2. 向量化
        self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_corpus)
        # 3. 分群
        self.df['Cluster_ID'] = self.clusterer.fit_predict(self.tfidf_matrix)
        
        print("訓練管線執行完畢！")

    def save_models(self, save_dir="models"):
        '''儲存模型到指定目錄'''
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 儲存工具狀態與資料
        joblib.dump(self.vectorizer.vectorizer, f"{save_dir}/tfidf_vectorizer.joblib")
        joblib.dump(self.clusterer.kmeans, f"{save_dir}/kmeans_model.joblib")
        joblib.dump(self.tfidf_matrix, f"{save_dir}/tfidf_matrix.joblib")
        self.df.to_pickle(f"{save_dir}/poker_data_clustered.pkl")
        print(f"資料與模型已安全儲存至 {save_dir}/")

# ==========================================
# 執行區塊
# ==========================================
if __name__ == "__main__":
    # 就像組裝積木一樣，把需要的工具實例化
    my_preprocessor = TextPreprocessor(dict_path="dict/custom_words.txt", stop_path="dict/stopwords.txt")
    my_vectorizer = VectorizationStrategy()
    my_clusterer = ClusteringStrategy(num_clusters=5)
    
    # 把工具交給訓練員
    trainer = PokerModelTrainer(my_preprocessor, my_vectorizer, my_clusterer)
    
    # 開始訓練並存檔
    trainer.train(excel_path="data/撲克資料.xlsx", text_column="欄位 A (text)")
    trainer.save_models()