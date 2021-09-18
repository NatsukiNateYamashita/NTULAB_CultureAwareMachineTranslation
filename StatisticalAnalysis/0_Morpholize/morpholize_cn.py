import os
import csv
import jieba
import jieba.posseg as pseg
from inlp.convert import chinese

def get_data(f_path):
    data = []
    with open(f_path, 'r', encoding='utf_8_sig') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            data.append(row[0])
    return data

def morpholize(data, elem_type):
    wakachi = []
    for i, d in enumerate(data):
        # print('traditional: ',d)
        d = chinese.t2s(d)
        # print('simplified:  ',d)
        if (elem_type == "morph") or (elem_type == "prim"):
            r = [word for word,_ in pseg.cut(d)]
            r = ' '.join(r)
        elif elem_type == "pos":
            r = [pos for _,pos in pseg.cut(d)]
            r = ' '.join(r)
        # print('morpholized:',r)
        wakachi.append([r])
        # if i >= 4:
        #     exit()
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
    name_dict = {   'mpdd/':['original_neg.csv','original_query.csv','original_res.csv'],
                    'cejc/':['rewrited_query.csv','rewrited_res.csv','translated_query.csv','translated_res.csv']}
    situation_list = ['request/','apology/','thanksgiving/']

    for corpus_dir, files in name_dict.items():
        for sit_dir in situation_list:
            os.makedirs(f'{save_dir}{corpus_dir}{sit_dir}', exist_ok=True)
            for fname in files:
                path = base_dir + corpus_dir + sit_dir + fname
                data = get_data(path)
                data = morpholize(data, elem_type)
                save_path = save_dir + corpus_dir + sit_dir + fname
                save_path = save_path[:-4]
                # print(save_path[:-4])
                # exit()
                save_data(save_path,data)