import torch
import math

class GraphConvolutionLayer(torch.nn.Module):
    def __init__(self,in_features,out_features,bias=True):
        super(GraphConvolutionLayer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = torch.nn.Linear(in_features,out_features)
        self.glorot(self.fc1.weight.data)
        self.zeros(self.fc1.bias.data)
        #torch.nn.init.xavier_normal_(self.fc1.weight.data)
        #torch.nn.init.normal_(self.fc1.bias.data)

    def glorot(self,tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0/(tensor.size(-2)+tensor.size(-1)))
            tensor.data.uniform_(-stdv,stdv)
    def zeros(self,tensor):
        if tensor is not None:
            tensor.data.fill_(0)



    def forward(self,input,adj):
        support = torch.mm(adj,input)
        output = self.fc1(support)
        return output



