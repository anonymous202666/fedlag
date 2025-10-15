import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch_geometric.nn import GCNConv, BatchNorm

 

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x @ x.T
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x ** 2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    if (x_sum == 0).any():
        x_sum[x_sum == 0] = 1e-8
 
    x_dis = x_dis * (x_sum ** (-1))
    del x_sum
    torch.cuda.empty_cache()
    x_dis = (1 - mask) * x_dis

    return x_dis


class GCN(nn.Module):
    def __init__(self, feat_dim, hid_dim, out_dim, dropout, num_layers=2):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.conv1 = GCNConv(feat_dim, hid_dim)
 
        
        if self.num_layers == 2:
            self.conv2 = GCNConv(hid_dim, hid_dim)

 
            self.fc3 = nn.Linear(hid_dim, hid_dim)   
            self.fc = nn.Linear(hid_dim, out_dim)  
    
        else:
    

            self.fc = nn.Linear(hid_dim, out_dim)   
        self.dropout = dropout

    def forward(self, data, return_x_dis=False,return_embed=False):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 2:
            # x = self.conv2(x, edge_index)
            x_hid = self.conv2((x), edge_index)
            
        else:
            x_hid = x 

        logit = self.fc(F.relu(x_hid) )
        if self.training:
            x_dis = get_feature_dis(F.relu(x_hid))
            return logit, x_dis
        elif return_x_dis:
            x_dis = get_feature_dis(F.relu(x_hid))
            return logit, x_dis
        elif return_embed:
            return logit, x_hid
        else:
 
            return logit

    def rep_forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.num_layers == 2:
            x = self.conv2(x, edge_index)   
        else:
            x = self.conv1(x, edge_index)   
        if self.training:
            x_dis = get_feature_dis(x)
            return x, x_dis
        else:
            return x

class FedTAD_ConGenerator(nn.Module):

    def __init__(self, noise_dim, feat_dim, out_dim, dropout):
        super(FedTAD_ConGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.emb_layer = nn.Embedding(out_dim, out_dim)

        hid_layers = []
        dims = [noise_dim + out_dim, 64, 128, 256]
        for i in range(len(dims) - 1):
            d_in = dims[i]
            d_out = dims[i + 1]
            hid_layers.append(nn.Linear(d_in, d_out))
            hid_layers.append(nn.Tanh())
            hid_layers.append(nn.Dropout(p=dropout, inplace=False))
        self.hid_layers = nn.Sequential(*hid_layers)
        self.nodes_layer = nn.Linear(256, feat_dim)

    def forward(self, z, c):
        z_c = torch.cat((self.emb_layer.forward(c), z), dim=-1)
        hid = self.hid_layers(z_c)
        node_logits = self.nodes_layer(hid)
        return node_logits
     
 
