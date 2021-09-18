from pyknp import Juman
jumanpp = Juman()

data = ["すもももももももものうち",
                            "井の中の蛙"]
wakachi = []
for d in data:
    # for mrph in result.mrph_list(): # 各形態素にアクセス
        # print("見出し:%s, 読み:%s, 原形:%s, 品詞:%s, 品詞細分類:%s, 活用型:%s, 活用形:%s, 意味情報:%s, 代表表記:%s" \
        #         % (mrph.midasi, mrph.yomi, mrph.genkei, mrph.hinsi, mrph.bunrui, mrph.katuyou1, mrph.katuyou2, mrph.imis, mrph.repname))
    r = [mrph.midasi for mrph in jumanpp.analysis(d).mrph_list()]
    r = " ".join(r)
    wakachi.append(r)
print(wakachi)

import jieba

data = ['真好吃耶', '不好意思我不對', '每天發技術文章']
wakachi = []
# 精確模式
for d in data:
    r = ' '.join(jieba.cut(d))
    wakachi.append(r)
    
print(wakachi)
# print('---------------')

# # 全模式
# for sentence in documents:
#     seg_list = jieba.cut(sentence, cut_all=True)
#     print(' '.join(seg_list))

# print('---------------')

# # 搜索引擎模式
# for sentence in documents:
#     seg_list = jieba.cut_for_search(sentence)
#     print('/'.join(seg_list))