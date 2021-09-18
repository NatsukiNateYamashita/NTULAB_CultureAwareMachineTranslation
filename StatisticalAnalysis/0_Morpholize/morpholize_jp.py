import os
import csv
from pyknp import Juman
jumanpp = Juman()

def get_data(f_path):
    data = []
    with open(f_path, 'r', encoding='utf_8_sig') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            data.append(row[0])
    return data

def morpholize(data, elem_type):
    wakachi = []
    for d in data:
        # for mrph in result.mrph_list(): # 各形態素にアクセス
            # print("見出し:%s, 読み:%s, 原形:%s, 品詞:%s, 品詞細分類:%s, 活用型:%s, 活用形:%s, 意味情報:%s, 代表表記:%s" \
            #         % (mrph.midasi, mrph.yomi, mrph.genkei, mrph.hinsi, mrph.bunrui, mrph.katuyou1, mrph.katuyou2, mrph.imis, mrph.repname))
        try:
            d=d.replace(" ","")
            if elem_type == "morph":
                r = [mrph.midasi for mrph in jumanpp.analysis(d).mrph_list()]
            elif elem_type == "prim":
                r = [mrph.genkei for mrph in jumanpp.analysis(d).mrph_list()]
            elif elem_type == "pos":
                r = [mrph.hinsi for mrph in jumanpp.analysis(d).mrph_list()]
        except:
            print(d)
            exit()
        r = " ".join(r)
        wakachi.append([r])
    return wakachi



def save_data(f_path,data):
    with open(f_path, 'w', encoding='utf_8_sig') as f:
        writer = csv.writer(f)
        writer.writerows(data)


if __name__ == "__main__":
    base_dir = "data/"
    save_dir = "mrphdata/"
    # save_dir = "primitive/"
    # save_dir = "pos/"
    elem_type = "morph"
    # elem_type = "prim"
    # elem_type = "pos"
    name_dict = {   'cejc/':['original_neg.csv','original_query.csv','original_res.csv'],
                    'mpdd/':['rewrited_query.csv','rewrited_res.csv','translated_query.csv','translated_res.csv']}
    situation_list = ['request/','apology/','thanksgiving/']

    for corpus_dir, files in name_dict.items():
        for sit_dir in situation_list:
            os.makedirs(f'{save_dir}{corpus_dir}{sit_dir}', exist_ok=True)
            for fname in files:
                path = base_dir + corpus_dir + sit_dir + fname
                data = get_data(path)
                data = morpholize(data, elem_type)
                print(data)
                # exit()
                save_path = save_dir + corpus_dir + sit_dir + fname
                save_path = save_path[:-4]
                
                save_data(save_path,data)
            


            
