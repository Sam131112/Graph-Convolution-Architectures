import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from my_utils import get_data,accuracy
from my_model import GCN,SimpleModel


np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

adj,features,labels,idx_train,idx_val,idx_test = get_data(True)


#model = GCN(nfeat=features.shape[1],nhid =256,nclass=labels.max().item()+1,dropout=0.5)
model = SimpleModel(nfeat=features.shape[1],nhid=256,nclass=labels.max().item()+1)
optimizer = optim.Adam(model.parameters(),lr = 0.01,weight_decay=5e-4)

model.cuda()

def train(epoch,x,adj,labels):
    optimizer.zero_grad()
    #output = model(x,adj)
    output = model(x)
    loss_train = F.nll_loss(output[idx_train],labels[idx_train])
    acc_train = accuracy(output[idx_train],labels[idx_train])
    loss_train.backward()
    optimizer.step()
    print("Epoch and Training Loss",epoch,loss_train.item())



def test(x,adj,labels):
    #output = model(x,adj)
    output = model(x)
    acc_test = accuracy(output[idx_test],labels[idx_test])
    print("Test set results:",acc_test.item())


for epoch in range(5000):
    train(epoch,features,adj,labels)

print("Optimization Finished!")

test(features,adj,labels)
