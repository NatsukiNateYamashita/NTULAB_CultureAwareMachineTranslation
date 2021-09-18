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

# 全てのインスタンスに対し、最もsemi-hardなネガティブサンプルを取得し、Rerutn

def semi_hard_negative_sample(net, qbatch, rbatch, nr, batch_size):
    alpha = 0.01
    batch_size = 128
    nbatch = []
    neg_idx = []
    for q,r in zip(qbatch,rbatch):
        # print(f'q.shape: {q.shape}')
        q = torch.from_numpy(q).float()
        r = torch.from_numpy(r).float()
        q = torch.unsqueeze(q,0)
        r = torch.unsqueeze(r,0)
        # print(f'q.shape: {q.shape}')
        # print(f'r.shape: {r.shape}')
        if torch.cuda.is_available():
            q,r = q.cuda(),r.cuda()
        with torch.no_grad(): 
            r_score = net(q, r)
        
        q=q.repeat(batch_size,1)
        idx = 0
        argmin = 0
        min_ = 999
        size = nr.shape[0]
        while True:
            n = nr[idx:idx+batch_size]
            if n.shape[0]!=batch_size:
                q = q[:n.shape[0]]
            if torch.cuda.is_available():
                n = torch.from_numpy(n).float()
                q,n = q.cuda(),n.cuda()
                # print(f'q.shape: {q.shape}')
                # print(f'n.shape: {n.shape}')
            ##########################ここの処理
            with torch.no_grad():
                n_score = net(q, n)
                neg_scores = r_score - n_score - alpha
                tmp_argmin = np.argmin(neg_scores.cpu().detach().numpy())
                tmp_min = neg_scores.min()
            if  min_ > tmp_min:
                min_ = tmp_min
                argmin = idx + tmp_argmin
            idx += batch_size
            if idx >= size:
                break

        neg_idx.append(argmin)
    neg_idx = np.array(neg_idx)
    nbatch = nr[neg_idx]
    return nbatch