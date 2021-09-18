import torch
import numpy as np
from pprint import pprint

class BERT_RUBER_refer():
    
    def __init__(self):
        self.bc = BertClient()
    
    def encode_sentence(self, sent):
        sent = [' '.join(i.split()[-200:]) for i in sent]
        return self.bc.encode(sent)    # [batch, 768]
    
    def encode_query(self, query):
        sentences = query.split('__eou__')
        se = self.bc.encode(sentences)
        return np.sum(se, axis=0)    # [768]
    
def cos_similarity(groundtruth, generated):
    try:
        sim = np.dot(groundtruth, generated) / (np.linalg.norm(groundtruth) * np.linalg.norm(generated))
    except ZeroDivisionError:
        sim = 0.0
    return sim

def calc_refer_score(translated_response , rewrited_response):
    sim = np.zeros(len(translated_response))
    for i,(t,r)in enumerate(zip(translated_response,rewrited_response)):
        sim[i] = cos_similarity(t,r)
    return sim
        

if __name__ == "__main__":
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained('bert-base-multilingual-uncased')
    translated_res = ["感謝いたします","誠にありがとうございます。"]
    rewroten_res = ["ありがとうございます","どうも"]
    t = tokenizer(translated_res, padding=True, truncation=True, return_tensors="pt")
    r = tokenizer(rewroten_res, padding=True, truncation=True, return_tensors="pt")
    model.eval()
    with torch.no_grad(): 
        t_outputs = model(**t,output_hidden_states=True,return_dict=True)
        r_outputs = model(**r,output_hidden_states=True,return_dict=True)
    t_all_encoder_layers = t_outputs.hidden_states
    r_all_encoder_layers = r_outputs.hidden_states
    t_embedding = t_all_encoder_layers[-2].detach().numpy()
    r_embedding = r_all_encoder_layers[-2].detach().numpy()
    # print('q_embedding.shape: ',t_embedding.shape)
    # print('r_embedding.shape: ',r_embedding.shape)
    t_embedding = np.max(t_embedding, axis=1)
    r_embedding = np.max(r_embedding, axis=1)
    # print('q_embedding.shape: ',t_embedding.shape)
    # print('r_embedding.shape: ',r_embedding.shape)
    scores = calc_refer_score(t_embedding,r_embedding)
    pprint('scores: {}'.format(scores))