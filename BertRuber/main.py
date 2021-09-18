import argparse
import random
import os
import sys
import time
import pprint
import csv
import math
from statistics import mean
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from pytorch_memlab import profile, MemReporter
from transformers import BertTokenizer, BertModel

from unreference_score import *
from reference_score import *
from train_unrefer import *


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
        net.load_state_dict(torch.load(file_path)['net'])
    else:
        raise Exception(f"[!] No saved model")

def get_scores(unref,ref):
    mean_unref = mean(unref)
    mean_ref = mean(ref)
    max_combination = []
    for u,r in zip(unref,ref):
        tmp = max(u,r)
        max_combination.append(tmp)
    mean_max = mean(max_combination)
    return mean_unref,mean_ref,mean_max


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tgt_lang = 'jp'
    corpus_dict = {'src':'cejc','tgt':'mpdd'} if tgt_lang == 'jp' else {'src':'mpdd','tgt':'cejc'}
    sit_list = ['request','apology','thanksgiving']
    # sit_list = ['apology','thanksgiving']

    val_class = {   
                    'translated':   {  'query':'translated',
                                        'res':'translated'},
                    'rewrited':     {   'query':'rewrited',
                                        'res':'rewrited'},
                    'tq2rw':        {   'query':'translated',
                                        'res':'rewrited'}
                }
    batch_size = 128
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained('bert-base-multilingual-uncased')


    for sit in sit_list:
        # get val data
        for key, items in val_class.items():
            path = 'data/{}/{}/{}_query.pkl'.format(corpus_dict['tgt'],sit,items['query'])
            val_query = get_pkldata(path) 
            path = 'data/{}/{}/{}_res.pkl'.format(corpus_dict['tgt'],sit,items['res']) 
            val_res = get_pkldata(path) 
            net = BERT_RUBER_unrefer(768, dropout=0.5)
            load_best_model(net, tgt_lang, sit)
            
            if torch.cuda.is_available():
                net.cuda()

            unref_scores = calc_unrefer_score(val_query, val_res, batch_size, net)
            # print(type(unref_scores))
            # print(len(unref_scores))
            # print(unref_scores)
            ref_scores = calc_refer_score(val_query, val_res)
            # print(type(ref_scores))
            # print(len(ref_scores))
            # print(ref_scores)
            mean_unref_score, mean_ref_score, mean_max = get_scores(unref_scores,ref_scores)
            print()
            print("tgt_lang: {}, situation: {}, combination: {}".format(tgt_lang,sit,key))
            print("mean_unref_score: {}, mean_ref_score: {}, mean_max: {}".format(mean_unref_score, mean_ref_score, mean_max))
            print()


