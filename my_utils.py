import pickle as pk
import numpy as np
import networkx as nx
import scipy as sp
import torch



#ind.cora.allx  ind.cora.ally  ind.cora.graph  ind.cora.test.index  ind.cora.tx  ind.cora.ty  ind.cora.x  ind.cora.y

def get_test_index(f):
    f1 = open(f,"r")
    ids = []
    for _f in f1:
        _f = _f.strip("\n\r")
        ids.append(int(_f))
    _t = np.array(ids)
    #_t = np.sort(_t)
    return _t


def convert_adj_to_sparse(adj):
    sparse_adj = sp.sparse.coo_matrix(adj,dtype=np.float32)
    indices = np.vstack((sparse_adj.row,sparse_adj.col))
    values = sparse_adj.data
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = sparse_adj.shape
    return torch.sparse.FloatTensor(i,v,torch.Size(s))



def get_data(cuda="True"):

    allx = pk.load(open("ind.cora.allx","rb"),encoding="latin1")
    ally = pk.load(open("ind.cora.ally","rb"),encoding="latin1")
    te_x = pk.load(open("ind.cora.tx","rb"),encoding="latin1")
    te_y = pk.load(open("ind.cora.ty","rb"),encoding="latin1")
    tr_x = pk.load(open("ind.cora.x","rb"),encoding="latin1")
    tr_y = pk.load(open("ind.cora.y","rb"),encoding="latin1")
    net = pk.load(open("ind.cora.graph","rb"),encoding="latin1")
    test_idx = get_test_index("ind.cora.test.index")
    test_idx_reorder = np.sort(test_idx)
    #print(tr_x.shape,te_x.shape,test_id.size,allx.shape)
    features = sp.sparse.vstack((allx,te_x)).tolil()  # because without lil format there is computational burden
    print(features.shape)
    features[test_idx,:] = features[test_idx_reorder,:]
    labels = np.vstack((ally,te_y))
    labels[test_idx,:] = labels[test_idx_reorder,:]
    idx_train = range(len(tr_y))
    idx_val = range(len(tr_y), len(tr_y)+500)
    g = nx.from_dict_of_lists(net)
    adj = nx.to_numpy_array(g,dtype=np.float)
    adj = adj + np.eye(adj.shape[0])
    #print(adj[5,2546],adj[2546,5])
    adj = sp.sparse.coo_matrix(adj)
    adj = normalize_adj(adj)
    features = normalize_features(features)
    adj = convert_adj_to_sparse(adj)  # this is converted to sparse format due to its immense size
    adj = torch.FloatTensor(adj.to_dense())
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    #print(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(test_idx_reorder)
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj,features,labels,idx_train,idx_val,idx_test
    #print(test_id)



def normalize_adj(m):
    rowsum = np.array(m.sum(1))
    rowsum = np.power(rowsum,-1).flatten()
    rowsum[np.isinf(rowsum)] = 0  # error handling
    D = sp.sparse.diags(rowsum)
    adj = D.dot(m).dot(D)     # D^-1*A'D^-1
    return adj


def normalize_features(m):
    rowsum = np.array(m.sum(1))
    rowsum = np.power(rowsum,-1).flatten()
    rowsum[np.isinf(rowsum)] = 0  # error handling
    D = sp.sparse.diags(rowsum)
    F = D.dot(m)     # D^-1*F
    return F


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



#adj,features,labels,idx_train,idx_val,idx_test = get_data()
#print("The Labels")
#print(labels)
