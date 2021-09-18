import torch
import torch.nn as nn
from torch.autograd import Function

from transformers import BertTokenizer, BertModel ,AdamW

class ReverseLayerF(Function):
    # https://arxiv.org/pdf/1505.07818.pdf
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class CustomBERTModelDANN(nn.Module):
    def __init__(self):
          super(CustomBERTModelDANN, self).__init__()
          self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
          self.linear1 = nn.Linear(768, 2) ## 2 is the number of classes in this example [Domain label]
          self.linear2 = nn.Linear(768, 2) ## 2 is the number of classes in this example (Class label)

    def forward(self, ids, mask, type_ids, alpha=1):
        _, pooled_output = self.bert(ids, attention_mask=mask, token_type_ids=type_ids, return_dict=False)
        y = ReverseLayerF.apply(pooled_output, alpha)  
        linear1_output = self.linear1(y) 
        linear2_output = self.linear2(pooled_output)
        return linear1_output, linear2_output
