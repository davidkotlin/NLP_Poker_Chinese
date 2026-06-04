import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# ==========================================
# 1. 系統初始化與資料載入
# ==========================================
print("⚙️ 系統啟動中：載入模型與字典...")

# 載入自訂字典 (讓系統看得懂你的查詢關鍵字)
jieba.load_userdict("dict/custom_words.txt")

# 讀取剛剛分好群的知識庫
df = pd.read_excel("data/分群結果.xlsx")
# 確保沒有空值，並轉成 List 方便運算
corpus = df['clean_text'].fillna("").astype(str).tolist()
raw_texts = df['欄位 A (text)'].fillna("").astype(str).tolist()

# 重建 TF-IDF 矩陣 (資料量小，瞬間就能算完)
vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
tfidf_matrix = vectorizer.fit_transform(corpus)

print(f"✅ 知識庫載入完成！共 {len(df)} 篇文章。")

# ==========================================
# 2. 定義混合搜尋引擎核心
# ==========================================
def search_poker_knowledge(query, top_k=5):
    # a. 處理使用者的輸入 (轉小寫並斷詞)
    query_clean = " ".join(jieba.lcut(query.lower()))
    
    # b. 計算 TF-IDF 語意相似度 (Cosine Similarity)
    query_vec = vectorizer.transform([query_clean])
    cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # c. 計算 RapidFuzz 容錯字串相似度 (Token Set Ratio 非常適合長文章比對)
    # 將 Fuzz 的 0~100 分數轉換成 0~1 的權重
    fuzz_scores = np.array([fuzz.token_set_ratio(query.lower(), text.lower()) / 100.0 for text in raw_texts])
    
    # d. 混合雙引擎分數 (60% 語意 + 40% 容錯比對)
    # 這樣既能懂概念，又能抓錯字
    final_scores = (0.6 * cosine_scores) + (0.4 * fuzz_scores)
    
    # e. 抓出分數最高的前 K 名
    # argsort() 會由小排到大，所以用 [::-1] 反轉，再取前 top_k 個
    top_indices = final_scores.argsort()[::-1][:top_k]
    
    return top_indices, final_scores

# ==========================================
# 3. 終端機互動介面 (CLI)
# ==========================================
print("\n" + "="*50)
print(" 🃏 歡迎來到德州撲克 AI 知識檢索系統 🃏")
print(" (輸入 'exit' 或 'q' 即可離開系統)")
print("="*50)

while True:
    user_input = input("\n🔍 請輸入你想查詢的撲克問題 (例如：怎麼算底池賠率？)：")
    
    if user_input.lower() in ['exit', 'q', 'quit']:
        print("系統關閉，下次見！👋")
        break
        
    if not user_input.strip():
        continue

    # 執行搜尋，預設回傳前 3 名
    indices, scores = search_poker_knowledge(user_input, top_k=3)
    
    print(f"\n為您找到最相關的 3 筆結果：")
    print("-" * 40)
    
    for rank, idx in enumerate(indices, 1):
        score = scores[idx]
        cluster_id = df['cluster'].iloc[idx]
        # 如果資料庫裡有來源欄位就抓，沒有的話就顯示預設文字
        source = df['欄位 B (source)'].iloc[idx] if '欄位 B (source)' in df.columns else "本地知識庫"
        text_snippet = str(df['欄位 A (text)'].iloc[idx])
        
        # 如果文章太長，只截取前 80 個字顯示
        if len(text_snippet) > 80:
            text_snippet = text_snippet[:80] + "..."
            
        print(f"🥇 第 {rank} 名 (綜合吻合度: {score:.2f}) | 標籤分類: 第 {cluster_id} 群")
        print(f"📖 來源: {source}")
        print(f"📝 內容: {text_snippet}")
        print("-" * 40)