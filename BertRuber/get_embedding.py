# import argparse
import os
import re
import csv
import glob

from transformers import BertTokenizer, BertModel
import torch
import pickle
import numpy as np

# def get_args():
#     parser = argparse.ArgumentParser(description='This is sample argparse script')
#     parser.add_argument('input_data_path',type=str, help='input_data_path')
#     parser.add_argument('output_data_path', type=int, help='output_data_path')

#     return parser.parse_args()

def get_data(f_path):
    data = []
    with open(f_path, 'r', encoding='utf_8_sig') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            data.append(row[0])
    return data

def get_embedding(data, tokenizer, model):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            model.cuda()
        # data = tokenizer(data, padding=True, truncation=True, return_tensors="pt")
        # data = data.to(device)
        input_ids = []
        attention_mask = []
        token_type_ids = []
        for d in data:
            encoded = tokenizer.encode_plus(d,
                                            max_length = 512,
                                            padding = 'max_length',
                                            truncation = True,
                                            add_special_tokens = True,
                                            pad_to_max_length = True,
                                            return_attention_mask = True)
            input_ids.append(encoded['input_ids'])
            attention_mask.append(encoded['attention_mask'])
            # token_type_ids.append(encoded['token_type_ids'])
        input_ids=torch.LongTensor(input_ids).to(device)
        attention_mask=torch.LongTensor(attention_mask).to(device)
        # token_type_ids=torch.LongTensor(token_type_ids)
        model.eval()
        with torch.no_grad(): 
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            # token_type_ids=token_type_ids,
                            output_hidden_states=True,
                            return_dict=True)
        embedding = outputs.hidden_states[-2].cpu().detach().numpy()
        embedding = np.max(embedding, axis=1)
        return embedding

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained('bert-base-multilingual-uncased')
    
    corpus_list = ['cejc','mpdd']
    for corpus in corpus_list:
        for input_fname in glob.glob('./data/{}/**/*.csv'.format(corpus), recursive=True):
            output_fname = os.path.splitext(input_fname)[0]
            output_fname += ".pkl"
            data = get_data(input_fname)
        # print(data[:2])
            embedding = []
            batch_size = 128
            idx = 0
            while True:
                batch = data[idx:idx+batch_size]
                idx += batch_size
                tmp = get_embedding(batch,tokenizer,model)
                embedding.extend(tmp.tolist())
                if idx >= len(data):
                    break
            embedding = np.array(embedding)
            print(embedding.shape)
            with open(output_fname, 'wb') as f:
                pickle.dump(embedding, f)
            print("saved {}".format(output_fname))

