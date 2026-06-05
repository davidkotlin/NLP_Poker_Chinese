'''一鍵補字腳本：把使用者會打的字補進對應文章，並新增一筆禮儀總覽。

做的事：
  1. 找到指定文章，在它「原本文字的結尾」接上一句話(不會新增列、不會產生重複)
  2. 新增一筆全新的「牌桌禮儀：新手禁忌總覽」
跑法：把這支放專案根目錄，執行  python patch_data.py
跑完記得重新訓練 formal_train_model.py，再重跑評測。
'''
import pandas as pd
import shutil

PATH  = "data/撲克資料.xlsx"
TEXT  = "欄位 A (text)"
SRC   = "欄位 B (source)"
LABEL = "欄位 C (my_label) 給你自己對答案用的"

shutil.copy(PATH, PATH + ".bak")          # 先備份，改壞了可以還原
df = pd.read_excel(PATH)

# (用來認出那篇的獨特關鍵字, 要接在結尾的句子)
edits = [
    ("開局加注該下多少", "也就是翻牌前你第一個加注時該加幾倍。"),                  # 第1題
    ("持續下注",         "也就是翻牌後你第一次下注(第一槍)要下多少。"),              # 第2題
    ("設定停損點",       "這樣才能避免一個晚上情緒一來就把錢全部輸光。"),            # 第4題
    ("二與四法則",       "也就是怎麼算你差一張牌(例如湊同花)的中獎機率。"),          # 第6題
    ("無法做出合理判斷", "例如輸牌之後情緒上來、一直想把錢討回來、開始亂打，這時最好先離桌冷靜。"),  # 第5題(上頭)
]

for key, sentence in edits:
    mask = df[TEXT].astype(str).str.contains(key, regex=False)
    n = int(mask.sum())
    if n == 0:
        print(f"⚠️  找不到含「{key}」的文章，請手動確認這篇還在不在")
        continue
    already = df[TEXT].astype(str).str.contains(sentence, regex=False)   # 跑過就不重複補
    todo = mask & ~already
    df.loc[todo, TEXT] = df.loc[todo, TEXT].astype(str).str.rstrip() + sentence
    note = "（找到多筆 → 有重複，建議之後用偵測器清掉）" if n > 1 else ""
    print(f"✅ 「{key}」：找到 {n} 筆，補上句子 {int(todo.sum())} 筆 {note}")

# 新增一筆全新的「禮儀總覽」(這是真的沒有，才用新增)
overview = ("新手在賭場牌桌上不該做的事(總覽)：輪到自己才行動，不要提前表態或亮牌；"
            "下注要一次推到位，不要分次推籌碼；不要 slow roll(贏了還故意拖著慢慢亮牌)；"
            "蓋牌後不要討論還在進行的牌局；保護好自己的底牌避免被荷官誤收；"
            "也不要催促或指導其他玩家。這些是牌桌最基本的禮儀禁忌。")
if not df[TEXT].astype(str).str.contains("不該做的事", regex=False).any():
    df = pd.concat([df, pd.DataFrame([{TEXT: overview, SRC: "新手補充(AI生成)",
                                       LABEL: "牌桌禮儀：新手禁忌總覽"}])], ignore_index=True)
    print("✅ 已新增一筆『牌桌禮儀：新手禁忌總覽』")
else:
    print("ℹ️  禮儀總覽已存在，略過")

df.to_excel(PATH, index=False)
print(f"\n完成！已存回 {PATH}（原檔備份在 {PATH}.bak）")
print("下一步：重新訓練 formal_train_model.py → 重跑 eval_recall_v2.py")