import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, LayerNorm
from torch.nn import Parameter


# adapted from https://github.com/chennnM/GBP
class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias='bn', bn_name='BatchNorm1d'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias == 'bn':
            self.bias = eval(bn_name)(int(out_features))
        else:
            self.bias = lambda x: x
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
    def forward(self, input):
        output = torch.mm(input, self.weight)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output


# MLP apply initial residual
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features,alpha,bns=False, bn_name='BatchNorm1d'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.alpha=alpha
        self.reset_parameters()
        self.bns=bns
        self.bias = eval(bn_name)(out_features)
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input ,h0):
        support = (1-self.alpha)*input+self.alpha*h0
        output = torch.mm(support, self.weight)
        #if self.bns:
        output=self.bias(output)
        if self.in_features==self.out_features:
            output = output+input
        return output


# adapted from dgl sign
class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout,bns=True, bn_name='BatchNorm1d'):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            self.bns.append(eval(bn_name)(hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
                self.bns.append(eval(bn_name)(hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.norm=bns
        self.reset_parameters()
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers -1: 
                if self.norm:
                    x = self.dropout(self.prelu(self.bns[layer_id](x)))
                else:
                    x = self.dropout(self.prelu(x))
        return x


class FeedForwardNetII(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout,alpha,bns=False,bn_name='BatchNorm1d'):
        super(FeedForwardNetII, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.in_feats=in_feats
        self.hidden=hidden
        self.out_feats=out_feats
        if n_layers == 1:
            self.layers.append(Dense(in_feats, out_feats,bn_name=bn_name))
        else:
            self.layers.append(Dense(in_feats, hidden,bn_name=bn_name))
            for i in range(n_layers - 2):
                self.layers.append(GraphConvolution(hidden, hidden,alpha,bns,bn_name=bn_name))
            self.layers.append(Dense(hidden, out_feats,bn_name=bn_name))

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
    def forward(self, x):
        x=self.layers[0](x)
        h0=x
        for layer_id, layer in enumerate(self.layers):
            if layer_id==0:
                continue
            elif layer_id== self.n_layers - 1:
                x = self.dropout(self.prelu(x))
                x = layer(x)
            else:
                x = self.dropout(self.prelu(x))
                x = layer(x,h0)
                #x = self.dropout(self.prelu(x))
        return x