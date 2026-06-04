'''TF-IDF 模型訓練與評估'''
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cluster import KMeans

# 1. 讀取前處理好的乾淨資料
print("正在載入乾淨的語料庫...")
df = pd.read_excel("data/clean_corpus.xlsx")

# 確保資料都是字串，如果有空值(NaN)就補成空字串
corpus = df['clean_text'].fillna("").astype(str).tolist()

# 2. 初始化 TF-IDF 模型 (Vectorizer, n. 名詞，向量化工具)
# 這裡加了兩個超實用的防呆參數：
# max_df=0.9 : 如果某個詞在 90% 以上的文章都出現，代表太氾濫，當作廢話踢掉。
# min_df=2   : 如果某個詞在全部 142 篇文章裡，出現不到 2 次(只出現1次)，當作罕見錯字踢掉。
vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)

# 3. 執行 TF-IDF 轉換：把文字變成數學矩陣 (Matrix, n. 名詞，矩陣)
tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

# ---------------------------------------------------------
# 以下是「檢驗區」：讓我們來偷看電腦的腦袋裡裝了什麼！
# ---------------------------------------------------------

print("\n=== 📊 TF-IDF 萃取報告 ===")
print(f"總文章數：{len(corpus)} 篇")
print(f"萃取出的有效特徵詞 (維度)：{len(feature_names)} 個")

print("\n🧐 隨機抽查 20 個特徵詞 (看看有沒有奇怪的雜訊)：")
# 隨機挑 20 個詞印出來看看
import random
random_features = random.sample(list(feature_names), min(20, len(feature_names)))
print(random_features)

print("\n🥇 讓我們看看『第一篇文章』的 TF-IDF 關鍵字：")
print(f"原始文章：{df['欄位 A (text)'].iloc[0][:50]}...") # 印出前50個字

# 抓出第一篇文章的數學分數陣列
first_doc_vector = tfidf_matrix[0].toarray()[0]

# 把「詞」跟「分數」綁在一起，並依照分數從高排到低
word_scores = [(feature_names[i], first_doc_vector[i]) for i in range(len(feature_names)) if first_doc_vector[i] > 0]
word_scores.sort(key=lambda x: x[1], reverse=True)

print("電腦認為這篇文章最重要的 5 個詞是：")
for word, score in word_scores[:5]:
    print(f" - {word}: 權重分數 {score:.4f}")
# ==========================================
# 4. 啟動 K-Means 分群演算法
# ==========================================

# 設定 K 值 (你要分成幾群？)
# 154 篇文章，我們先試著分成 5 群 (例如: 牌型規則、下注流程、策略數學、特殊名詞 等)
num_clusters = 5
print(f"\n🚀 開始進行 K-Means 分群 (設定 K={num_clusters})...")

# 初始化 K-Means 模型
# n_init=10 代表讓電腦隨機丟 10 次中心點，選最好的一次，避免運氣差分得不好
# random_state=42 是為了讓你每次跑的結果都一樣，方便除錯
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)

# 訓練模型並取得分群標籤 (0, 1, 2, 3, 4)
kmeans.fit(tfidf_matrix)

# 把電腦算出來的群組編號 (cluster)，當作一個新欄位塞回我們原本的 DataFrame
df['cluster'] = kmeans.labels_

# ==========================================
# 5. 檢驗成果：電腦到底把什麼文章分在同一群？
# ==========================================
print("\n=== 🏷️ 各群組的核心關鍵字 ===")
# 抓出每一群的「中心點」在 484 個維度中的分數排序
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    # 計算這一群有幾篇文章
    cluster_size = sum(df['cluster'] == i)
    print(f"\n📍 第 {i} 群 (共 {cluster_size} 篇文章):")
    
    # 抓出這一群分數最高的前 5 個代表字
    top_words = [feature_names[ind] for ind in order_centroids[i, :5]]
    print("核心特徵詞：", ", ".join(top_words))

# ==========================================
# 6. 將最終結果匯出成 Excel
# ==========================================
print("\n正在將分群結果匯出至 data 資料夾...")

# 為了方便你看 Excel，我們把 'cluster' 欄位移到最前面
cols = ['cluster', '欄位 A (text)', 'clean_text', '欄位 B (source)', '欄位 C (my_label)']
# 把如果有遺漏的欄位補上
final_cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]

df[final_cols].to_excel("data/分群結果.xlsx", index=False)
print("匯出完成！請打開 data/分群結果.xlsx，看看電腦的分群跟你的 my_label 像不像！")