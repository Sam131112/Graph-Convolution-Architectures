import torch
import torch.nn as nn
import torch.nn.functional as F
from my_layer import GraphConvolutionLayer


class GCN(nn.Module):
    def __init__(self,nfeat,nhid,nclass,dropout):
        super(GCN,self).__init__()
        self.gcn1 = GraphConvolutionLayer(nfeat,nhid)
        self.gcn2 = GraphConvolutionLayer(nhid,nhid)
        self.fc1 = nn.Linear(nhid,nclass)
        torch.nn.init.xavier_normal_(self.fc1.weight.data)
        torch.nn.init.normal_(self.fc1.bias.data)
        self.dropout = dropout

    def forward(self,x,adj):
        x = F.relu(self.gcn1(x,adj))
        x = F.dropout(x,self.dropout)
        x = self.gcn2(x,adj)
        x = self.fc1(x)
        return F.log_softmax(x,dim=1)



