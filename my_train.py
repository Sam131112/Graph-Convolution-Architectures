import torch
import time
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from my_utils import get_data,accuracy,get_data_v1
from my_model import GCN,SimpleModel


np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

torch.cuda.set_device(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

g,adj,features,labels,idx_train,idx_val,idx_test = get_data_v1(True)

#adj1,features1,labels1,idx_train1,idx_val1,idx_test1 = get_data(True)
#print(type(adj1),type(adj))
#_a = adj - adj1
#print("Checks " ,torch.sum(_a).item())
#print(idx_train.size())
#print("Features")
#print(features)

model = GCN(nfeat=features.shape[1],nhid =256,nclass=labels.max().item()+1,dropout=0.5)
#model = SimpleModel(nfeat=features.shape[1],nhid=256,nclass=labels.max().item()+1)
optimizer = optim.Adam(model.parameters(),lr = 0.01,weight_decay=5e-4)

model.cuda()

patience = 500


def train(epoch,x,adj,labels):
    optimizer.zero_grad()
    output = model(x,adj)
    #output = model(x)
    loss_train = F.nll_loss(output[idx_train],labels[idx_train])
    #print("Model Out ",output[idx_train])
    #print("Model Out ",labels[idx_train])
    acc_train = accuracy(output[idx_train],labels[idx_train])
    #print(epoch,acc_train.item())
    loss_train.backward()
    optimizer.step()
    return loss_train.item()
    #print("Epoch and Training Loss",epoch,loss_train.item())



def test(x,adj,labels):
    output = model(x,adj)
    #output = model(x)
    acc_test = accuracy(output[idx_test],labels[idx_test])
    print("Test set results:",acc_test.item())


best = 1e10
patience_now = 0
curr_t = time.time()

for epoch in range(100):
    train_acc = train(epoch,features,adj,labels)
    #print(epoch,train_acc)
    output = model(features,adj)
    val_loss = F.nll_loss(output[idx_train],labels[idx_train])
    if val_loss.item() < best:
        best = val_loss.item()
        torch.save(model.state_dict(),"saved_model.pth")
        patience_now = 0
    patience_now+=1
    '''
    if patience_now>=patience:
        print("Train loss, Validation loss :",train_acc,best)
        break
    '''

print("elapsd time ",(time.time()-curr_t)/60.0)

print("Optimization Finished!")
#model = GCN(nfeat=features.shape[1],nhid =256,nclass=labels.max().item()+1,dropout=0.5)
#device = torch.device('cuda')
#model.load_state_dict(torch.load("saved_model.pth"))
#model.cuda()
model.eval()
test(features,adj,labels)
