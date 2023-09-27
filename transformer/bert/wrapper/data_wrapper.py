# data wrapper
# data loader wrapper
# gnn wrapper
import torch


class PygLinkDataWrapper():
    def __init__(self, data, split_edge):
        self.data = data
        self.split_edge = split_edge
    
    @property
    def num_nodes(self,):
        return self.data.num_nodes
    
    @property
    def y(self,):
        return None
    
    @property
    def train_mask(self,):
        return None
    
    @property
    def val_mask(self,):
        return None
    
    @property
    def test_mask(self,):
        return None
    
    def permute(self, perm):
        inv_perm = torch.zeros_like(perm)
        inv_perm[perm] = torch.arange(len(perm))
        for split in self.split_edge:
            for key in self.split_edge[split]:
                self.split_edge[split][key] = inv_perm[self.split_edge[split][key]]

class DglLinkDataWrapper():
    def __init__(self, data, split_edge):
        self.data = data
        self.split_edge = split_edge
    
    @property
    def num_edges(self,):
        return self.data.num_edges()
    
    @property
    def num_nodes(self,):
        return self.data.number_of_nodes()
    
    @property
    def y(self,):
        return None
    
    @property
    def train_mask(self,):
        return None
    
    @property
    def val_mask(self,):
        return None
    
    @property
    def test_mask(self,):
        return None
    
    def permute(self, perm):
        inv_perm = torch.zeros_like(perm)
        inv_perm[perm] = torch.arange(len(perm))
        for split in self.split_edge:
            for key in self.split_edge[split]:
                self.split_edge[split][key] = inv_perm[self.split_edge[split][key]]


class PygNodeDataWrapper():
    def __init__(self, data):
        self.data = data

    @property
    def num_nodes(self,):
        return self.data.num_nodes
    
    @property
    def y(self,):
        return self.data.y
    
    @property
    def train_mask(self,):
        return self.data.train_mask
    
    @property
    def val_mask(self,):
        return self.data.val_mask
    
    @property
    def test_mask(self,):
        return self.data.test_mask
        

class DglDataWrapper():
    def __init__(self, graph):
        self.data = graph

    @property
    def num_nodes(self,):
        return self.data.number_of_nodes()
    
    @property
    def y(self,):
        return self.data.ndata['labels']
    
    @property
    def train_mask(self,):
        return self.data.ndata['train_mask']
    
    @property
    def val_mask(self,):
        return self.data.ndata['val_mask']
    
    @property
    def test_mask(self,):
        return self.data.ndata['test_mask']