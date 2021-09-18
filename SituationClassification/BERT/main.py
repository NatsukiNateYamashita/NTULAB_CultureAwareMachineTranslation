
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np

from statistics import mean
import argparse
import random
import os
import sys
import time
import pprint
import csv
import math
from tqdm import tqdm
from pytorch_memlab import profile, MemReporter
from transformers import BertTokenizer, BertModel

from train import *



class Config():
    def __init__(self):
        # self.dropout = 0.5
        # self.weight_decay=1e-4
        # self.lr=1e-3
        # self.epoches = 200
        # self.grad_clip = 10
        self.batch_size = 4
        # self.tgt_lang = 'jp'
        # self.reverse = True    


def load_best_model(net, tgt_lang, sit):
    import torch
    path = f"./ckpt/{tgt_lang}/{sit}/"
    best_acc, best_file = -1, None
    best_epoch = -1
    
    for file in os.listdir(path):
        try:
            _, acc, _, loss, _, epoch = file.split("_")
            epoch = epoch.split('.')[0]
        except:
            continue
        acc = float(acc)
        epoch = int(epoch)
        if epoch > best_epoch:
        # if acc > best_acc:
            best_file = file
            best_epoch = epoch
            # best_acc = acc

    if best_file:
        file_path = path + best_file
        # print(f'[!] Load the model from {file_path}')
        net.load_state_dict(torch.load(file_path)['model'])
    else:
        raise Exception(f"[!] No saved model")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tgt_lang_list = ['jp','cn','mix'] 
    for tgt_lang in tgt_lang_list:
        if tgt_lang == 'jp':
            corpus_dict = {'src':'cejc','tgt':'mpdd'} 
        elif tgt_lang == 'cn':
            corpus_dict = {'src':'mpdd','tgt':'cejc'}
        else:
            corpus_dict = {'src':'mix','tgt':'cejc'}
            corpus_dict = {'src':'mix','tgt':'mpdd'}
        
        sit_list = ['request','apology','thanksgiving']
        # sit_list = ['apology','thanksgiving']
        # sit_list = ['thanksgiving']

        val_class = {   
                        'translated':   {  'query':'translated',
                                            'res':'translated'},
                        'rewrited':     {   'query':'rewrited',
                                            'res':'rewrited'},
                        'tq2rw':        {   'query':'translated',
                                            'res':'rewrited'}
                    }
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        model = BertModel.from_pretrained('bert-base-multilingual-uncased')
        conf = Config()
        for sit in sit_list:

            # get val data
            for key, items in val_class.items():
                # print()
                # print("tag_lang: {}, sit: {}, class: {}".format(tgt_lang,sit,items['query']))
                data_dir = '/nfs/nas-7.1/yamashita/LAB/dialoguedata/'
                val_query_path = data_dir + '{}/{}/{}_res.csv'.format(corpus_dict['tgt'], sit, items['query'])
                val_res_path = data_dir + '{}/{}/{}_res.csv'.format(corpus_dict['tgt'], sit, items['res'])
                val_query = get_data(val_query_path)
                val_res = get_data(val_res_path)

                net = CustomBERTModelDANN()
                load_best_model(net, tgt_lang, sit)
                
                if torch.cuda.is_available():
                    net.cuda()

                val_iter = get_val_batch(val_query,val_res,conf.batch_size)
                _, _, scores = validation(val_iter,net)
                
                scores = np.array(scores)
                mean_scores = np.mean(np.max(scores, axis=1))
                variance = np.var(np.max(scores, axis=1))
                acc = np.sum(np.argmax(scores, axis=1)) / len(val_res)

                # print()
                print("tgt_lang: {}, situation: {}, combination: {}".format(tgt_lang,sit,key))
                print("mean_scores: {}, variance: {}, acc: {}".format(mean_scores, variance, acc))
                print()


