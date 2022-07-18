import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraModel, RobertaModel
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel


# KLUE, Soongsil
class HateSpeechBERTModel(nn.Module):
    def __init__(self, bert, num_class):
        super().__init__()
        self.bert = bert
        self.dropout1 = nn.Dropout(p=0.3)
        self.fcl = nn.Linear(768, 1024)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1024, num_class)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = x.pooler_output
        x = self.dropout1(x)
        x = self.fcl(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        out = self.softmax(x)
        return out


# KcElectra, KoElectra
class HateSpeechELECTRAModel(nn.Module):
    def __init__(self, bert, num_class):
        super().__init__()
        self.bert = bert
        self.dropout1 = nn.Dropout(p=0.3)
        self.fcl = nn.Linear(768, 1024)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1024, num_class)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = x.last_hidden_state[:, 0, :]
        x = self.dropout1(x)
        x = self.fcl(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        out = self.softmax(x)
        return out
