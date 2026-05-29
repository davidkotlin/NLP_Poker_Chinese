'''處理資歷，比對專有名詞字典，jieba切句子'''
import pandas as pd
import jieba

# 1. 載入自訂字典
# 讓 Jieba 認識德撲黑話，確保「翻牌前」不會被切成「翻牌 / 前」
jieba.load_userdict("dict/custom_words.txt")

# 2. 載入停用詞表（黑名單）
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        # 讀取每一行，並去掉換行符號，只保留非空白的詞
        return set([line.strip() for line in f if line.strip()])

stopwords = load_stopwords("dict/stopwords.txt")
print(f"成功載入 {len(stopwords)} 個停用詞！")

# 3. 讀取 Excel 原始資料
# 這裡使用 pandas 讀取 xlsx 檔案
print("正在讀取資料...")
df = pd.read_excel("data/撲克資料.xlsx")

# 檢查一下有沒有成功讀到前幾筆
print(f"成功讀取 {len(df)} 筆資料！")
# print("前三筆資料長這樣：\n", df.head(3))

# 4. 修改後的斷詞函式：加入停用詞過濾機制
def cut_and_filter_text(text):
    # 先將傳入的文字強制轉為字串，並全部轉為小寫，避免大小寫出入導致字典比對錯誤
    text = str(text).lower()
    
    # 使用 jieba 進行切詞
    raw_tokens = jieba.lcut(text)
    
    clean_tokens = []
    for token in raw_tokens:
        token = token.strip()
        # 條件：確保詞彙不是空白、長度大於 0，且「不在停用詞黑名單內」
        if token and token not in stopwords:
            clean_tokens.append(token)
            
    # 用空白鍵把過濾後的乾淨詞彙接起來
    return " ".join(clean_tokens)

# 5. 執行斷詞 (Preprocess, v. 動詞，前處理)
# 對 'text' 這個欄位的每一列資料，套用剛剛寫好的斷詞函式，並存成一個新欄位 'clean_text'
print("\n開始進行 Jieba 斷詞...")
df['clean_text'] = df['欄位 A (text)'].apply(cut_and_filter_text)

# 6. 檢視斷詞結果
print("\n正在將斷詞結果匯出至 data 資料夾...")

# 直接使用 pandas 的 .to_excel() 匯出完整結果
# df[['欄位 A (text)', 'clean_text']].to_excel("test/斷詞結果檢查.xlsx", index=False)
# print("匯出完成！請打開 test/斷詞結果檢查.xlsx 查看完整內容。")
# 6. (可選) 將清理好的資料存下來，供下一個程式使用
df[['欄位 A (text)', 'clean_text']].to_excel("data/clean_corpus.xlsx", index=False)