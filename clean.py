import pandas as pd
TEXT = "欄位 A (text)"; LABEL = "欄位 C (my_label) 給你自己對答案用的"
df = pd.read_excel("data/撲克資料.xlsx")
print("總筆數:", len(df))
dup = df[df.duplicated(subset=[TEXT], keep=False)].sort_values(TEXT)
print("重複文字筆數:", len(dup))
print(dup[[TEXT, LABEL]].to_string())

df = df.drop_duplicates(subset=[TEXT]).reset_index(drop=True)   # 留一份就好
df.to_excel("data/撲克資料.xlsx", index=False)
print("去重後:", len(df))