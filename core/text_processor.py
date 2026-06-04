'''文字處理模組'''
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