import argparse
import random
import os
import sys
import time
import pprint
import csv
import math
from tqdm import tqdm
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from unreference_score import *
from negative_sample import*

# set the random seed for the model
random.seed(20)
torch.manual_seed(20)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(20)

def get_pkldata(f_path):
    with open(f_path, mode='rb') as f:
        data = pickle.load(f)
    return data

def get_train_batch(net, q, pr, nr, batch_size):
    if len(q) != len(pr):
        print('the length of query and respose is different!')
        exit()
    # print('len(q): {}, len(pr): {}'.format(len(q),len(pr)))
    size = len(q)
    idx = 0
    while True:
        qbatch = q[idx:idx+batch_size]
        rbatch = pr[idx:idx+batch_size]

        ####### semi hard negative sample
        # nbatch = semi_hard_negative_sample(net, qbatch, rbatch, nr, batch_size)
        ####### random
        pidx = np.random.choice(nr.shape[0], batch_size)
        nbatch = nr[pidx]
        #######

        qbatch = np.concatenate([qbatch, qbatch])
        rbatch = np.concatenate([rbatch, nbatch])
        
        label = np.concatenate([np.ones(int(qbatch.shape[0] / 2)),
                                np.zeros(int(qbatch.shape[0] / 2))])

        # shuffle
        pureidx = np.arange(qbatch.shape[0])
        np.random.shuffle(pureidx)
        qbatch = qbatch[pureidx]
        rbatch = rbatch[pureidx]
        label = label[pureidx]
        
        idx += batch_size
        yield qbatch, rbatch, label
        
        if idx >= size:
            break
    return None 

def get_val_batch(q, r, batch_size):

    if len(q) != len(r):
        print('the length of query and respose is different!')
        exit()    
    size = len(q)
    idx = 0
    while True:
        qbatch = q[idx:idx+batch_size]
        rbatch = r[idx:idx+batch_size]
      
        label = np.concatenate([np.ones(int(qbatch.shape[0]))])

        pureidx = np.arange(qbatch.shape[0])
        np.random.shuffle(pureidx)
        qbatch = qbatch[pureidx]
        rbatch = rbatch[pureidx]
        label = label[pureidx]

        idx += batch_size
        yield qbatch, rbatch, label
        
        if idx >= size:
            break
    return None 
    
def train(data_iter, net, optimizer, grad_clip=10):
    net.train()
    batch_num, losses = 0, 0
    criterion = nn.BCELoss()
    
    for batch_idx, batch in enumerate(data_iter):
        qbatch, rbatch, label = batch
        qbatch = torch.from_numpy(qbatch).float()
        rbatch = torch.from_numpy(rbatch).float()
        label = torch.from_numpy(label).float()
        batch_size = qbatch.shape[0]
        
        if torch.cuda.is_available():
            qbatch, rbatch = qbatch.cuda(), rbatch.cuda()
            label = label.cuda()
            
        optimizer.zero_grad()
        scores = net(qbatch, rbatch)
        loss = criterion(scores, label)

        loss.backward()
        clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        losses += loss.item()
        batch_num = batch_idx + 1
    return round(losses / batch_num, 4)

def validation(data_iter, net):
    net.eval()
    losses, batch_num, acc, acc_num = 0, 0, 0, 0
    all_score = []
    criterion = nn.BCELoss()
    
    for batch_idx, batch in enumerate(data_iter):
        qbatch, rbatch, label = batch 
        qbatch = torch.from_numpy(qbatch).float()
        rbatch = torch.from_numpy(rbatch).float()
        label = torch.from_numpy(label).float()
        batch_size = qbatch.shape[0]
                
        if torch.cuda.is_available():
            qbatch, rbatch = qbatch.cuda(), rbatch.cuda()
            label = label.cuda()
            
        scores = net(qbatch, rbatch)
        all_score.extend(scores.cpu().detach().tolist())
        loss = criterion(scores, label)
        
        s = scores >= 0.5
        acc += torch.sum(s.float() == label).item()
        acc_num += batch_size
        batch_num += 1
        losses += loss.item()
        
    return round(losses / batch_num, 4), round(acc / acc_num, 4), all_score

def calc_unrefer_score(query, response, batch_size, net):
    val_iter = get_val_batch(query, response, batch_size)
    _,_,scores = validation(val_iter, net)

    return scores


if __name__ == "__main__":
    from pytorch_memlab import profile, MemReporter
    from transformers import BertTokenizer, BertModel

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tgt_lang = 'jp'
    corpus_dict = {'src':'cejc','tgt':'mpdd'} if tgt_lang == 'jp' else {'src':'mpdd','tgt':'cejc'}
    sit_list = ['request','apology','thanksgiving']
    # sit_list = ['request']

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained('bert-base-multilingual-uncased')
    for sit in sit_list:
        print(f'tgt-lang: {tgt_lang}, {sit}')
        # get train data
        path = 'data/{}/{}/original_query.pkl'.format(corpus_dict['src'],sit)
        train_query = get_pkldata(path)
        path = 'data/{}/{}/original_res.pkl'.format(corpus_dict['src'],sit)
        train_pos = get_pkldata(path)
        path = 'data/{}/{}/original_neg.pkl'.format(corpus_dict['src'],sit)
        train_neg = get_pkldata(path)

        # get val data
        path = 'data/{}/{}/rewrited_query.pkl'.format(corpus_dict['tgt'],sit)
        val_query = get_pkldata(path) 
        path = 'data/{}/{}/rewrited_res.pkl'.format(corpus_dict['tgt'],sit) 
        val_res = get_pkldata(path) 
    
        dropout = 0.5
        dim = 768
        weight_decay=1e-4
        lr=1e-5
        epoches, grad_clip = 200, 10
        batch_size = 8

        net = BERT_RUBER_unrefer(dim, dropout=dropout)
        if torch.cuda.is_available():
            net.cuda()
        optimizer = optim.Adam( net.parameters(), 
                                lr=lr
                                # weight_decay=weight_decay,
                                )
        
        training_losses, validation_losses, validation_metrices = [], [], []
        min_loss = np.inf
        best_metric = -1
        os.makedirs(f'ckpt/{tgt_lang}/{sit}/', exist_ok=True)
        os.system(f'rm ckpt/{tgt_lang}/{sit}/*')
        print(f'[!] Clear the checkpoints under ckpt')
        patience = 0
        # pbar = tqdm(range(1, epoches + 1))
        begin_time = time.time()
        # for epoch in pbar:
        for epoch in range(1, epoches + 1):
            train_iter = get_train_batch(net, train_query, train_pos, train_neg, batch_size)
            val_iter = get_val_batch(val_query, val_res, batch_size)
            # test_iter = get_batch(testqpath, testrpath, 256)
            training_loss = train(train_iter, net, optimizer)
            validation_loss, validation_metric, _ = validation(val_iter, net)
            training_losses.append(training_loss)
            validation_losses.append(validation_loss)
            validation_metrices.append(validation_metric)
            if best_metric < validation_metric:
                patience = 0
                best_metric = validation_metric
                min_loss = validation_loss
            else:
                patience += 1
                
            state = {'net': net.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                    'epoch': epoch}
            
            if epoch > 15:
                torch.save(state,
                    f'ckpt/{tgt_lang}/{sit}/Acc_{validation_metric}_vloss_{validation_loss}_epoch_{epoch}.pt')
            print('training_loss: {}, validation_loss: {}, validation_metric: {}, patience: {}'.format(training_loss,validation_loss,validation_metric, patience))
            # pbar.set_description(f"loss(train-dev): {training_loss}-{validation_loss}, Acc: {validation_metric}, patience: {patience}")
            if patience > 15:
                print(f'[!] early stop')
                break
        # pbar.close()
            
        # calculate costing time
        end_time = time.time()
        hour = math.floor((end_time - begin_time) / 3600)
        minute = math.floor(((end_time - begin_time) - 3600 * hour) / 60)
        second = (end_time - begin_time) - hour * 3600 - minute * 60
        print(f"Cost {hour}h, {minute}m, {round(second, 2)}s")
        
        scores = calc_unrefer_score(val_query, val_res, batch_size, net)
        print(scores[:5])
        print()
