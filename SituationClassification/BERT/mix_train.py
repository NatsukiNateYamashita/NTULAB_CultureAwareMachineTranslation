import os
import time
import csv
import math
import gc

from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from transformers import BertTokenizer, BertModel ,AdamW
from model import *

class Config():
    def __init__(self):
        self.dropout = 0.5
        self.weight_decay=1e-4
        self.lr=5e-3
        self.epoches = 200
        self.grad_clip = 10
        self.batch_size = 4
        self.tgt_lang = 'jp'
        self.reverse = True
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')


def get_data(f_path):
    data = []
    with open(f_path, 'r', encoding='utf_8_sig') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            data.append(row[0])
    return data

def encoding(conf,q, r):
    if len(q)==1:
        encoded = conf.tokenizer.encode_plus(q[0],r[0],
                                        max_length = 128,
                                        truncation = True,
                                        padding = 'max_length',
                                        add_special_tokens = True,
                                        pad_to_max_length=True,
                                        return_attention_mask=True
                                        )
        input_ids=torch.unsqueeze(torch.LongTensor(encoded['input_ids']),0)
        attention_mask=torch.unsqueeze(torch.LongTensor(encoded['attention_mask']),0)
        token_type_ids=torch.unsqueeze(torch.LongTensor(encoded['token_type_ids']),0)
        
    else:
        input_ids=[]
        attention_mask=[]
        token_type_ids=[]
        for qq, rr in zip(q,r):
            encoded = conf.tokenizer.encode_plus(qq,rr,
                                            max_length = 128,
                                            truncation = True,
                                            padding = 'max_length',
                                            add_special_tokens = True,
                                            pad_to_max_length=True,
                                            return_attention_mask=True
                                            )
            input_ids.append(encoded['input_ids'])
            attention_mask.append(encoded['attention_mask'])
            token_type_ids.append(encoded['token_type_ids'])
        input_ids= torch.LongTensor(input_ids)
        attention_mask= torch.LongTensor(attention_mask)
        token_type_ids= torch.LongTensor(token_type_ids)
    return input_ids, attention_mask, token_type_ids

def negative_sample(conf,net, qbatch, rbatch, nr, batch_size):
    neg_alpha = 0.01
    batch_size = 128
    nbatch = []
    neg_idx = []
    q_tmp = []
    r_tmp = []
    for q,r in zip(qbatch,rbatch):
        q_tmp.append(q)
        r_tmp.append(r)
        input_ids, attention_mask, token_type_ids = encoding(conf,q_tmp,r_tmp)
        input_ids = torch.unsqueeze(input_ids,0)
        attention_mask = torch.unsqueeze(attention_mask,0)
        token_type_ids = torch.unsqueeze(token_type_ids,0)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
            # print('input_ids.shape:', input_ids.shape)
        _, r_score = net(input_ids,attention_mask,token_type_ids)
        for i in range(batch_size-1):
            q_tmp.append(q)
        # print('q type:     ',type(q))
        # print('q_tmp type: ',type(q_tmp))
        # print('q_tmp len:  ',len(q_tmp))
        # print(q_tmp)

        idx = 0
        argmin = 0
        min_ = 999
        size = len(nr)
        while True:
            n = nr[idx:idx+batch_size]
            # print('len(n): ',len(n))
            if len(n)!=batch_size:
                q_tmp = q_tmp[:len(n)]
            input_ids, attention_mask, token_type_ids = encoding(q_tmp,n)
            # print('input_ids.shape: ',input_ids.shape)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                # print(f'q.shape: {q.shape}')
                # print(f'n.shape: {n.shape}')
            ##########################???????????????
            with torch.no_grad():
                _, n_score = net(input_ids,attention_mask,token_type_ids)
                # print('len(n)       : ',len(n))
                # print('len(q_tmp)       : ',len(q_tmp))
                # print('r_score.shape: ',r_score.shape)
                # print('n_score.shape: ',n_score.shape)
                neg_scores = abs(r_score - n_score - neg_alpha)
                tmp_argmin = np.argmin(neg_scores.cpu().detach().numpy())
                tmp_min = neg_scores.min()
            if  min_ > tmp_min:
                min_ = tmp_min
                argmin = idx + tmp_argmin
                print('argmin: ',argmin)
                print('min: ',tmp_min)
            idx += batch_size
            print('idx: ', idx)
            if idx >= size:
                break

        neg_idx.append(argmin)
    neg_idx = np.array(neg_idx)
    nbatch = nr[neg_idx]
    return nbatch


def get_train_batch(conf, model, j_q, j_pr, j_nr, c_q, c_pr, c_nr):
    if len(j_q) != len(j_pr):
        print('the length of query and respose is different!')
        exit()
    if len(c_q) != len(c_pr):
        print('the length of query and respose is different!')
        exit()
    j_size = len(j_q)
    c_size = len(c_q)
    j_idx = 0
    c_idx = 0
    batch_divide = 24
    j_batch_size = len(j_q) // batch_divide
    c_batch_size = len(c_q) // batch_divide
    while True:
        j_qbatch = j_q[j_idx:j_idx+j_batch_size]
        j_rbatch = j_pr[j_idx:j_idx+j_batch_size]
        pidx = np.random.choice(len(j_nr), j_batch_size)
        j_nr = np.array(j_nr)
        j_nbatch = j_nr[pidx]
        j_qbatch = np.concatenate([j_qbatch, j_qbatch])
        j_rbatch = np.concatenate([j_rbatch, j_nbatch])
        j_sit_label = np.concatenate([np.ones(int(j_qbatch.shape[0] / 2)),
                                np.zeros(int(j_qbatch.shape[0] / 2))])    
        j_lang_label = np.zeros(j_qbatch.shape[0])
        j_lang_reversed_label = np.ones(j_qbatch.shape[0])

        c_qbatch = c_q[c_idx:c_idx+c_batch_size]
        c_rbatch = c_pr[c_idx:c_idx+c_batch_size]
        pidx = np.random.choice(len(c_nr), c_batch_size)
        c_nr = np.array(c_nr)
        c_nbatch = c_nr[pidx]
        c_qbatch = np.concatenate([c_qbatch, c_qbatch])
        c_rbatch = np.concatenate([c_rbatch, c_nbatch])
        c_sit_label = np.concatenate([np.ones(int(c_qbatch.shape[0] / 2)),
                                np.zeros(int(c_qbatch.shape[0] / 2))])    
        
        c_lang_label = np.zeros(c_qbatch.shape[0])
        c_lang_reversed_label = np.ones(c_qbatch.shape[0])

        q_batch = np.concatenate([j_qbatch, c_qbatch])
        r_batch = np.concatenate([j_rbatch, c_rbatch])
        sit_label = np.concatenate([j_sit_label, c_sit_label])
        lang_label = np.concatenate([j_lang_label, c_lang_label])
        lang_reversed_label = np.concatenate([j_lang_reversed_label, c_lang_reversed_label])   
                    # print('qbatch: {}, rbatch: {}, label: {}'.format(qbatch.shape,rbatch.shape,label.shape))
        # print('label: ',label)
        # shuffle
        pureidx = np.arange(q_batch.shape[0])
        np.random.shuffle(pureidx)
        q_batch = q_batch[pureidx]
        r_batch = r_batch[pureidx]
        sit_label = sit_label[pureidx]
        lang_label = lang_label[pureidx]
        lang_reversed_label = lang_reversed_label[pureidx]
        j_idx += j_batch_size
        c_idx += c_batch_size
        del c_nbatch
        del c_qbatch
        del c_rbatch
        del c_sit_label
        del c_lang_label
        del c_lang_reversed_label
        del j_nbatch
        del j_qbatch
        del j_rbatch
        del j_sit_label
        del j_lang_label
        del j_lang_reversed_label
        gc.collect()   
        yield q_batch, r_batch, sit_label, lang_label, lang_reversed_label
        
        if j_idx >= j_size or c_idx >= c_size :
            break
    del j_q, j_pr, j_nr, c_q, c_pr, c_nr
    gc.collect()    
    return None 

def get_val_batch(j_q, j_r, c_q, c_r, batch_size):
    if len(j_q) != len(j_r) or len(c_q) != len(c_r):
        print('the length of query and respose is different!')
        exit()    
    q = np.concatenate([j_q, c_q])
    r = np.concatenate([j_r, c_r])
    size=q.shape[0]
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
    del j_q, j_r, c_q, c_r
    gc.collect()   
    return None 

def train(conf, model, optimizer, data_iter, epoch, data_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    train_lang_loss = 0
    train_situ_loss = 0
    train_loss = 0
    # input_ids=[]
    # attention_mask=[]
    # token_type_ids=[]
    for batch_idx, batch in enumerate(data_iter):
        qbatch, rbatch, sit_label, lang_label, lang_reversed_label = batch
        
        i=batch_idx*len(qbatch)
        p = float(i + epoch * data_size / epoch / data_size)
        dann_alpha = 2. / (1. + np.exp(-10 * p)) - 1
        input_ids, attention_mask, token_type_ids = encoding(conf,qbatch,rbatch)
        sit_label = torch.LongTensor(sit_label)
        lang_label = torch.LongTensor(lang_label)
        lang_reversed_label = torch.LongTensor(lang_reversed_label)
        del batch,qbatch, rbatch
        gc.collect()
        if torch.cuda.is_available():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            sit_label = sit_label.to(device)
            lang_label = lang_label.to(device) if conf.reverse == False else lang_reversed_label.to(device)
   
        optimizer.zero_grad()
        output_1, output_2  = model(input_ids,
                                    attention_mask,
                                    token_type_ids,
                                    dann_alpha)
        lang_loss = F.cross_entropy(output_1,lang_label)
        situ_loss = F.cross_entropy(output_2,sit_label)

        loss = lang_loss + situ_loss
        loss.backward()
        clip_grad_norm_(model.parameters(), conf.grad_clip)
        optimizer.step()
        train_lang_loss += lang_loss
        train_situ_loss += situ_loss
        train_loss  += loss.item()
        batch_num = batch_idx + 1
    if train_lang_loss <= 0.03:
        conf.reverse = not conf.reverse
    train_lang_loss=train_lang_loss.cpu().detach().numpy()
    train_situ_loss=train_situ_loss.cpu().detach().numpy()
    print("\nTrainingEpoch...: {}, train_loss: {}, train_situ_loss: {}, train_lang_loss: {}".format(epoch,round(train_loss / batch_num, 4),round(train_situ_loss / batch_num, 4),round(train_lang_loss / batch_num, 4)))
    return round(train_situ_loss / batch_num, 4)

def validation(conf, data_iter, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs_list = []
    all_score = []
    acc = 0
    acc_num = 0
    val_loss = 0
    model.eval()

    # input_ids=[]
    # attention_mask=[]
    # token_type_ids=[]

    for batch_idx, batch in enumerate(data_iter):
        qbatch, rbatch, sit_label = batch
        input_ids, attention_mask, token_type_ids = encoding(conf,qbatch,rbatch)
        
        sit_label = torch.LongTensor(sit_label)
        batch_size = input_ids.shape[0]
        if torch.cuda.is_available():
            # qbatch, rbatch = qbatch.cuda(), rbatch.cuda()
            # label = label.cuda()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            sit_label = sit_label.to(device)

        with torch.no_grad():
            scores = model(input_ids,attention_mask,token_type_ids)
            scores = scores[1]

            # lang_loss = F.cross_entropy(output_1,lang_label)
            sit_loss = F.cross_entropy(scores,sit_label)
            all_score.extend(scores.cpu().detach().tolist())

            # s = scores >= 0.5
            scores = torch.argmax(scores.cpu().detach(), dim=1)
            # print("type(s): {} s: {}".format(type(scores),scores))
            # print("scores.shape: {} sit_label.shape: {}".format(scores.shape,sit_label.shape))
            acc += torch.sum(scores.float() == sit_label.cpu().detach()).item()
            acc_num += batch_size

            for b in input_ids.cpu().numpy():
                temp = conf.tokenizer.convert_ids_to_tokens([x for x in b])
                temp = ''.join(temp)
                temp = temp.replace('##','')
                temp = temp.replace('[PAD]','')
                inputs_list.append(temp)
            val_loss += sit_loss
            batch_num = batch_idx + 1
    val_loss=val_loss.cpu().detach().numpy()
    all_score=np.array(all_score)
    return round(val_loss / batch_num, 4), round(acc / acc_num, 4), all_score
        



if __name__ == "__main__":
    conf = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # tgt_lang_list = ['jp','cn'] 
    # tgt_lang_list = ['cn'] 
    sit_list = ['request','apology','thanksgiving']
    sit_list = ['apology','thanksgiving']
    sit_list = ['thanksgiving']
    for sit in sit_list:
        data_dir = '/nfs/nas-7.1/yamashita/LAB/BertRuber/data/'
        j_train_query_path = data_dir + 'cejc/{}/original_query.csv'.format(sit)
        j_train_res_path = data_dir + 'cejc/{}/original_res.csv'.format(sit)
        j_train_neg_path = data_dir + 'cejc/{}/original_neg.csv'.format(sit)
        c_val_query_path = data_dir + 'mpdd/{}/rewrited_res.csv'.format(sit)
        c_val_res_path = data_dir + 'mpdd/{}/rewrited_res.csv'.format(sit)
        j_train_query = get_data(j_train_query_path)
        j_train_res = get_data(j_train_res_path)
        j_train_neg = get_data(j_train_neg_path)
        c_val_query = get_data(c_val_query_path)
        c_val_res = get_data(c_val_res_path)
        c_train_query_path = data_dir + 'mpdd/{}/original_query.csv'.format(sit)
        c_train_res_path = data_dir + 'mpdd/{}/original_res.csv'.format(sit)
        c_train_neg_path = data_dir + 'mpdd/{}/original_neg.csv'.format(sit)
        j_val_query_path = data_dir + 'cejc/{}/rewrited_res.csv'.format(sit)
        j_val_res_path = data_dir + 'cejc/{}/rewrited_res.csv'.format(sit)
        c_train_query = get_data(c_train_query_path)
        c_train_res = get_data(c_train_res_path)
        c_train_neg = get_data(c_train_neg_path)
        j_val_query = get_data(j_val_query_path)
        j_val_res = get_data(j_val_res_path)


        model = CustomBERTModelDANN()
        optimizer = AdamW(model.parameters(), 
                                lr=conf.lr
                                # weight_decay=weight_decay,
                                )
        if torch.cuda.is_available():
            model.cuda()

        data_size = len(j_train_query)+len(c_train_query)
        training_losses, validation_losses, validation_metrices = [], [], []
        min_loss = np.inf
        best_metric = -1
        os.makedirs(f'ckpt/mix/{sit}/', exist_ok=True)
        os.system(f'rm ckpt/mix/{sit}/*')
        print(f'[!] Clear the checkpoints under ckpt')
        patience = 0
        begin_time = time.time()


        for epoch in range(1, conf.epoches+1):  
            train_iter = get_train_batch(conf,model,j_train_query,j_train_res,j_train_neg,c_train_query,c_train_res,c_train_neg)
            val_iter = get_val_batch(j_val_query,j_val_res,c_val_query,c_val_res,conf.batch_size)

            training_loss = train(conf, model, optimizer, train_iter, epoch, data_size)
            validation_loss, validation_metric, _ = validation(conf, val_iter, model)

            training_losses.append(training_loss)
            validation_losses.append(validation_loss)
            validation_metrices.append(validation_metric)
            
            if best_metric < validation_metric:
                patience = 0
                best_metric = validation_metric
                min_loss = validation_loss
            else:
                patience += 1
                
            state = {'model': model.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                    'epoch': epoch}
            if epoch > 3:
                torch.save(state,
                    f'ckpt/mix/{sit}/Acc_{validation_metric}_vloss_{validation_loss}_epoch_{epoch}.pt')
            print('training_loss: {}, validation_loss: {}, validation_metric: {}, patience: {}'.format(training_loss,validation_loss,validation_metric, patience))
        # pbar.set_description(f"loss(train-dev): {training_loss}-{validation_loss}, Acc: {validation_metric}, patience: {patience}")
            if patience > 15:
                print(f'[!] early stop')
                break
        

        
    end_time = time.time()
    hour = math.floor((end_time - begin_time) / 3600)
    minute = math.floor(((end_time - begin_time) - 3600 * hour) / 60)
    second = (end_time - begin_time) - hour * 3600 - minute * 60
    print(f"Cost {hour}h, {minute}m, {round(second, 2)}s")
    
    print()



