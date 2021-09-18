
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

def get_data(f_path):
    data = []
    with open(f_path, 'r', encoding='utf_8_sig') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            data.append(row[0])

    return data


class BERT_RUBER_unrefer(nn.Module):
    
    def __init__(self,embedding_size, dropout=0.5):
        super(BERT_RUBER_unrefer, self).__init__()
        
        self.M = nn.Parameter(torch.rand(embedding_size, embedding_size))
        self.layer1 = nn.Linear(embedding_size * 2 + 1, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 128)
        self.opt = nn.Linear(128, 1)
        self.opt_drop = nn.Dropout(p=dropout)
        
    def forward(self, query, reply):
        # query / replty: 768-dim bert embedding
        # [B, H]
        qh = query.unsqueeze(1)    # [B, 1, 768]
        rh = reply.unsqueeze(2)    # [B, 768, 1]
        score = torch.bmm(torch.matmul(qh, self.M), rh).squeeze(2)  # [B, 1]
        qh = qh.squeeze(1)    # [B, H]
        rh = rh.squeeze(2)    # [B, H]
        linear = torch.cat([qh, score, rh], 1)    # [B, 2 * H  + 1]
        linear = torch.tanh(self.layer1(linear))
        linear = torch.tanh(self.layer2(linear))
        linear = torch.tanh(self.layer3(linear))
        linear = torch.sigmoid(self.opt(linear).squeeze(1))  # [B]
        
        return linear


if __name__ == "__main__":
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained('bert-base-multilingual-uncased')
    q = "今日は来てくれてありがと"
    r = "いやいや、暇だったから"
    q = tokenizer(q, return_tensors="pt")
    r = tokenizer(r, return_tensors="pt")
    model.eval()
    with torch.no_grad(): 
        q_outputs = model(**q,output_hidden_states=True,return_dict=True)
        r_outputs = model(**r,output_hidden_states=True,return_dict=True)
    q_all_encoder_layers = q_outputs.hidden_states
    r_all_encoder_layers = r_outputs.hidden_states
    q_embedding = q_all_encoder_layers[-2].detach().numpy()[0]
    r_embedding = r_all_encoder_layers[-2].detach().numpy()[0]
    print('q_embedding.shape: ',q_embedding.shape)
    print('r_embedding.shape: ',r_embedding.shape)
    
    qt = np.max(q_embedding, axis=0)
    rt = np.max(r_embedding, axis=0)
    # qt = np.mean(q_embedding, axis=0)
    # rt = np.mean(r_embedding, axis=0)
    print('qt.shape: ',qt.shape)
    print('rt.shape: ',rt.shape)
    qt = qt.reshape(1, 768)
    rt = rt.reshape(1, 768)
    print('qt.shape: ',qt.shape)
    print('rt.shape: ',rt.shape)
   