'''kmeans分群處理'''
import numpy as np
import joblib
from sklearn.cluster import KMeans
import os
# ==========================================
# 模組 3：專職處理分群 (KMeans)
# ==========================================
class ClusteringStrategy:
    '''這個類別專門負責分群邏輯，完美封裝 sklearn 與 joblib 的細節'''
    
    def __init__(self, num_clusters=5):
        # 預設建立一個全新的 KMeans 機器人
        self.kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        
    def fit_predict(self, matrix):
        '''【訓練用】訓練模型並直接回傳所有資料的標籤
        這fit_predict方法與kmeans.fit_predict不同
        用它來封裝kmeans.fit_predict
        讓PokerModelTrainer的主訓練區train可以多形
        ==========================================
        fit()完後還要再labels_取得分群標籤
        fit_predict()會同時做完fit()和labels_，直接回傳分群標籤
        '''
        print(f"執行 KMeans 分群 (K={self.kmeans.n_clusters})...")
        return self.kmeans.fit_predict(matrix)
        
    # def predict_cluster(self, vector):
    #     '''【搜尋用】預測單一向量屬於哪一群'''
    #     # 這裡把討人厭的 [0] 細節給封裝起來了！
    #     # 主控台呼叫時，只會拿到一個乾淨的整數 (例如 3)
    #     return self.kmeans.predict(vector)[0]
    def predict_top_k_clusters(self, vector, top_k=2):
        '''【進階搜尋用】計算距離，並回傳最接近的前 K 個群組 ID'''
        
        # transform 會回傳該向量到「每個群組中心點」的距離 (例如: [0.8, 0.2, 0.9, 0.3, 0.7])
        # [0] 是為了把結果從二維陣列剝離出來
        distances = self.kmeans.transform(vector)[0]
        
        # argsort 會由小排到大。因為這是「距離」，越小代表越接近！
        # 所以我們直接取陣列的「最前面 top_k 個」
        closest_clusters = np.argsort(distances)[:top_k]
        
        # 轉成標準的 Python List 回傳 (例如: [1, 3])
        return closest_clusters.tolist()

    def save_model(self, filepath):
        '''把肚子裡的 kmeans 機器人存進硬碟(存取參數以利之後預測使用)'''
        joblib.dump(self.kmeans, filepath)
        
    def load_model(self, filepath):
        '''從硬碟喚醒 kmeans 機器人，取代原本空的機器人(取得存取的參數)'''
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到分群模型檔案：{filepath}，請先訓練模型！")
        self.kmeans = joblib.load(filepath)