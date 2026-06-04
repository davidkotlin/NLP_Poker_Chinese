'''TF-IDF 相關的「訓練、搜尋、存檔、讀檔」還有「矩陣本身」'''
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os
# ==========================================
# 模組 2：專職處理向量化 (TF-IDF)
# ==========================================
class VectorizationStrategy:
    '''這個類別專門負責向量化邏輯，讓主控台不用管 sklearn 與存檔細節'''
    
    def __init__(self):
        # 初始化機器人
        self.vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
        # 把算出來的矩陣也存在自己肚子裡，主控台不用幫忙拿
        self.tfidf_matrix = None 

    def fit_transform(self, cleaned_corpus):
        '''【訓練用】計算 TF-IDF 矩陣並記在肚子裡
        這fit_transform方法與vectorizer.fit_transform不同
        用它來封裝vectorizer.fit_transform
        讓PokerModelTrainer的主訓練區train可以多形
        '''
        print("計算 TF-IDF 矩陣...")
        self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_corpus)
        return self.tfidf_matrix

    def transform(self, cleaned_query):
        '''【搜尋用】將新句子轉換成向量 (只套用，不學習)'''
        return self.vectorizer.transform(cleaned_query)

    def save_model(self, vec_path, mat_path):
        '''把肚子裡的字典跟矩陣存進硬碟(存取參數以利之後預測使用)'''
        joblib.dump(self.vectorizer, vec_path)
        joblib.dump(self.tfidf_matrix, mat_path)
        
    def load_model(self, vec_path, mat_path):
        '''從硬碟喚醒字典與矩陣'''
        if not os.path.exists(vec_path) or not os.path.exists(mat_path):
            raise FileNotFoundError("找不到向量化模型或矩陣檔案，請先訓練模型！")
        self.vectorizer = joblib.load(vec_path)
        self.tfidf_matrix = joblib.load(mat_path)