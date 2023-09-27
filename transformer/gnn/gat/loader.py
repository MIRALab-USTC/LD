import time
import torch

from dgl.dataloading import DataLoader
from dgl.dataloading import MultiLayerFullNeighborSampler, MultiLayerNeighborSampler

from bert.bert_gnn_sampler import RecursiveSampler, OneshotSampler

def get_gat_loader(data, conf):

    all_idx = torch.arange(data.num_nodes)
    train_idx = all_idx[data.data.ndata['train_mask']]
    # train_batch_size = lm_batch_size # (len(train_idx) + 9) // 10

    # graph = data.data.cpu().clone()
    # graph.ndata['train_mask'] = graph.ndata['train_mask'].long()
    # graph.ndata.pop('val_mask')
    # graph.ndata.pop('test_mask')
    
    # batch_size = len(train_idx)
    train_sampler = MultiLayerNeighborSampler([32 for _ in range(conf.model.params.architecture.n_layers)])

    eval_sampler = MultiLayerNeighborSampler([100 for _ in range(conf.model.params.architecture.n_layers)])
    # sampler = MultiLayerFullNeighborSampler(conf.model.params.architecture.n_layers)
    eval_dataloader = DataLoader(
            data.data.cpu(),
            all_idx,
            eval_sampler,
            batch_size=65536,
            num_workers=10,
        )
    
    train_loader_func = lambda gnn_batch_size: DataLoader(
            data.data.cpu(),
            train_idx.cpu(),
            train_sampler,
            batch_size=gnn_batch_size,
            shuffle=True,
            num_workers=10,
        )
    
    
    return data, train_loader_func, eval_dataloader