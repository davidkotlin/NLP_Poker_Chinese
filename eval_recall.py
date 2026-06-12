'''評測腳本：逐題印出前三名，由你「人眼判讀」算 recall@3

為什麼不用自動 snippet 比對：
  獨立出的題目（出題者沒看過文章原文）寫的正解句子，
  幾乎不會逐字出現在文章內文裡，模糊比對會把明明對的判成 ❌，跑出假的低分。
  對這種題庫，最可靠的判讀就是人眼看前三名。

前置：models/ 已有訓練好的檔；題庫 xlsx 放專案根目錄，至少要有 query 欄。
      （若還有 reference_answer / note / category 欄，會順手印出來幫你判斷。）
執行：python eval_recall.py
'''
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from core.text_processor import TextPreprocessor
from core.vectorizer import VectorizationStrategy
from core.embedding import EmbeddingStrategy, reciprocal_rank_fusion

MODEL_DIR = "models"
TEXT_COL = "欄位 A (text)"
LABEL_COL = "欄位 C (my_label) 給你自己對答案用的"
EVAL_FILE = "data/撲克隱藏題庫.xlsx"
TOP_K = 3

# --- 載入跟搜尋引擎一樣的工具 ---
pre = TextPreprocessor("dict/custom_words.txt", "dict/stopwords.txt")
vec = VectorizationStrategy()
vec.load_model(f"{MODEL_DIR}/tfidf_vectorizer.joblib", f"{MODEL_DIR}/tfidf_matrix.joblib")
emb = EmbeddingStrategy()
emb.load_model(f"{MODEL_DIR}/doc_embeddings.joblib")
df = pd.read_pickle(f"{MODEL_DIR}/poker_data_clustered.pkl")


def search_top(query, k=TOP_K):
    '''回傳前 k 名的 (內文, label, 語意信心)
    註：這是 hybrid(TF-IDF + embedding 用 RRF 融合) 版本。
    如果你本機已經把 search_top 換成接了 reranker 的版本，保留你的那段就好，
    這個檔我只動了「判讀」的部分。'''
    cleaned = pre.clean([query])[0]
    tfidf_scores = cosine_similarity(vec.transform([cleaned]), vec.tfidf_matrix).flatten()
    emb_scores = emb.similarity(emb.transform([query]))
    final = reciprocal_rank_fusion([tfidf_scores, emb_scores])
    idx = final.argsort()[-k:][::-1]
    return [(str(df.iloc[i][TEXT_COL]), df.iloc[i][LABEL_COL], round(float(emb_scores[i]), 3))
            for i in idx]


def run():
    evalset = pd.read_excel(EVAL_FILE)
    n = len(evalset)
    marks = []   # 1 = 命中, 0 = 沒中, None = 略過
    print(f"共 {n} 題。看完每題前三名後，輸入 y(命中) / n(沒中) / s(略過)\n")

    for i, r in evalset.iterrows():
        print("=" * 60)
        print(f"[{i + 1}/{n}] 問題：{r['query']}")
        for col in ("reference_answer", "note", "category"):   # 有就印出來幫你判
            if col in evalset.columns and pd.notna(r.get(col)):
                print(f"   參考（{col}）：{str(r[col])[:]}")
        for rank, (content, lab, conf) in enumerate(search_top(r["query"]), 1):
            print(f"   {rank}. [{lab}] (信心 {conf})  {content[:]}…")

        ans = input("   前三名有命中嗎？ y / n / s ＞ ").strip().lower()
        marks.append(1 if ans == "y" else (None if ans == "s" else 0))
        print()

    judged = [m for m in marks if m is not None]
    skipped = marks.count(None)
    hit = sum(judged)

    print("=" * 60)
    if judged:
        print(f"recall@{TOP_K} = {hit}/{len(judged)} = {hit / len(judged) * 100:.1f}%"
              f"   （人工判讀{('，略過 ' + str(skipped) + ' 題') if skipped else ''}）")
    else:
        print("沒有判讀任何一題。")

    misses = [evalset.iloc[i]["query"] for i, m in enumerate(marks) if m == 0]
    if misses:
        print("\n沒中的題目（優先檢查：是資料根本沒有，還是被擠掉）：")
        for q in misses:
            print("  -", q)

    out = evalset.copy()
    out["judge"] = ["命中" if m == 1 else ("略過" if m is None else "沒中") for m in marks]
    out.to_excel("eval_judged.xlsx", index=False)
    print("\n判讀結果已存到 eval_judged.xlsx（下次想回顧不用重判）")


if __name__ == "__main__":
    run()
