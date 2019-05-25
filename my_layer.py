import torch


class GraphConvolutionLayer(torch.nn.Module):
    def __init__(self,in_features,out_features,bias=True):
        super(GraphConvolutionLayer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = torch.nn.Linear(in_features,out_features)
        torch.nn.init.xavier_normal_(self.fc1.weight.data)
        torch.nn.init.normal_(self.fc1.bias.data)


    def forward(self,input,adj):
        support = torch.mm(adj,input)
        output = self.fc1(support)
        return output



