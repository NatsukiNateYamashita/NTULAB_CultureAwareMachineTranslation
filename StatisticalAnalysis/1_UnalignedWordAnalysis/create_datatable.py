
import os
import argparse
import pandas as pd
from collections import OrderedDict
from collections import defaultdict
import pprint
import json
import csv
class relations():
    def __init__(self):
        with open('metadata.json') as f:
            d = json.load(f)
        self.dict = d
        
    def convert(self, val):
        if val in self.dict['position']['inferior']:
            return 'inferior'
        if val in self.dict['position']['peer']:
            return 'peer'
        if val in self.dict['position']['superior']:
            return 'superior'
    def get(self,corpus,situation,upper=True):
        relation_pair = []
        relation_path = f'../relation_pair/{corpus}/{situation}/relation_pair'
        
        with open(relation_path, 'r', encoding='utf-8-sig')as f:
            reader = csv.reader(f)
            for line in reader:
                if upper == True:
                    line[0] = self.convert(line[0])
                    line[1] = self.convert(line[1])
                relation_pair.append(line)
        return relation_pair


def get_data_as_list(fpath) -> list:
    data = []
    with open(fpath, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.replace("\n","").split(" ")
            data.append(line)
    return data

if __name__ == "__main__":
    corpora = ["cejc","mpdd"]
    situations = ["request","apology","thanksgiving"]
    methods = ["rewrited","translated"]
    sen_types = ["query", "res"]
    diff_types = ["del","add"]
    upperRelation = relations()

    table = []
    table_upper = []
    for corpus in corpora:
        for situation in situations:
            for method in methods:
                for typ in sen_types:
                    for diff in diff_types:
                        ref = "original" if diff == "del" else method
                            
                        unaligned_index_fpath= 'unaligned_index/' + f"{corpus}/{situation}/{method}_{typ}.{diff}"
                        mrph_fpath=            '../mrphdata/'     + f"{corpus}/{situation}/{ref}_{typ}"
                        prim_fpath=            '../primitive/'    + f"{corpus}/{situation}/{ref}_{typ}"
                        pos_fpath =            '../pos/'          + f"{corpus}/{situation}/{ref}_{typ}"
                        relation_fpath =       '../relation_pair/'+ f"{corpus}/{situation}/relation_pair"

                    # get mrph, prim and pos data\
                        mrph = get_data_as_list(mrph_fpath)
                        prim = get_data_as_list(prim_fpath)
                        pos = get_data_as_list(pos_fpath)
                        rel = upperRelation.get(corpus,situation,upper=False)
                        upper_rel = upperRelation.get(corpus,situation,upper=True)
                        rel_col = 0 if typ == 'query' else 1
                    # get line, index and every correspond data
                        unaligned_index = get_data_as_list(unaligned_index_fpath)
                        for i, index in enumerate(unaligned_index):
                            for indice in index:
                                if indice == "":
                                    continue
                                if pos[int(i)][int(indice)] in ['x', '特殊', '助詞', '未定義語']: # x:非语素词 包含标点符号
                                    continue


                                table.append([corpus, situation, method, typ, diff, rel[int(i)][rel_col], upper_rel[int(i)][rel_col], i, indice, mrph[int(i)][int(indice)], prim[int(i)][int(indice)], pos[int(i)][int(indice)]])
                                try:
                                    if (((corpus == 'cejc') and (diff =='add')) or ((corpus == 'mpdd') and (diff == 'del'))):
                                        table_upper.append([corpus, situation, method, typ, diff, rel[int(i)][rel_col], upper_rel[int(i)][rel_col], i, indice, mrph[int(i)][int(indice)], prim[int(i)][int(indice)], pos[int(i)][int(indice)][0]])
                                    else:
                                        table_upper.append([corpus, situation, method, typ, diff, rel[int(i)][rel_col], upper_rel[int(i)][rel_col], i, indice, mrph[int(i)][int(indice)], prim[int(i)][int(indice)], pos[int(i)][int(indice)]])
                                except:
                                    print(pos[int(i)][int(indice)])
                    # save as csv
    column_name=['corpus', 'situation', 'method', 'sentence type', 'difference type', 'relation', 'upper relation', 'line', 'index', 'word', 'primitive form', 'pos']   
    df = pd.DataFrame(table,columns=column_name)
    df.to_csv('analysis_table.csv', encoding='utf-8-sig')
    df_upper = pd.DataFrame(table_upper,columns=column_name)
    df_upper.to_csv('analysis_table_upper.csv', encoding='utf-8-sig')
