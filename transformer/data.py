from typing import Tuple

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import (Planetoid, WikiCS, Coauthor, Amazon,
                                      GNNBenchmarkDataset, Yelp, Flickr,
                                      Reddit2, PPI)
import ogb
from ogb.nodeproppred import PygNodePropPredDataset, DglNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset, DglLinkPropPredDataset

from bert.wrapper.data_wrapper import PygNodeDataWrapper, DglDataWrapper, PygLinkDataWrapper, DglLinkDataWrapper

def index2mask(idx: torch.Tensor, size: int) -> torch.Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask

def get_arxiv_dgl(root: str):
    dataset = DglNodePropPredDataset(root=f'{root}/OGB', name='ogbn-arxiv')

    graph, labels = dataset[0]
    split_idx = dataset.get_idx_split()
    graph.ndata["labels"] = labels[:,0]

    evaluator = ogb.nodeproppred.Evaluator(name='ogbn-arxiv')
    evaluator_wrapper = lambda pred, labels: evaluator.eval({"y_pred": pred[:,None], "y_true": labels[:,None]})

    graph.ndata['train_mask'] = index2mask(split_idx['train'], graph.number_of_nodes())
    graph.ndata['val_mask'] = index2mask(split_idx['valid'], graph.number_of_nodes())
    graph.ndata['test_mask'] = index2mask(split_idx['test'], graph.number_of_nodes())

    return DglDataWrapper(graph), graph.ndata["feat"].shape[-1], labels[:,0].max().item()+1, evaluator_wrapper, "acc"

def get_proteins_dgl(root: str):
    dataset = DglNodePropPredDataset(root=f'{root}/OGB', name='ogbn-proteins')

    graph, labels = dataset[0]
    split_idx = dataset.get_idx_split()
    graph.ndata["labels"] = labels.float()

    evaluator = ogb.nodeproppred.Evaluator(name='ogbn-proteins')
    evaluator_wrapper = lambda pred, labels: evaluator.eval({"y_pred": pred, "y_true": labels})

    graph.ndata.pop('species')

    graph.ndata['train_mask'] = index2mask(split_idx['train'], graph.number_of_nodes())
    graph.ndata['val_mask'] = index2mask(split_idx['valid'], graph.number_of_nodes())
    graph.ndata['test_mask'] = index2mask(split_idx['test'], graph.number_of_nodes())
    
    return DglDataWrapper(graph), 0, labels.shape[1], evaluator_wrapper, "rocauc"

def get_products_dgl(root: str):
    dataset = DglNodePropPredDataset(root=f'{root}/OGB', name='ogbn-products')

    graph, labels = dataset[0]
    split_idx = dataset.get_idx_split()
    graph.ndata["labels"] = labels[:,0]

    evaluator = ogb.nodeproppred.Evaluator(name='ogbn-products')
    evaluator_wrapper = lambda pred, labels: evaluator.eval({"y_pred": pred[:,None], "y_true": labels[:,None]})

    graph.ndata['train_mask'] = index2mask(split_idx['train'], graph.number_of_nodes())
    graph.ndata['val_mask'] = index2mask(split_idx['valid'], graph.number_of_nodes())
    graph.ndata['test_mask'] = index2mask(split_idx['test'], graph.number_of_nodes())
    
    return DglDataWrapper(graph), graph.ndata["feat"].shape[-1], labels[:,0].max()+1, evaluator_wrapper, "acc"


def get_proteins_pyg(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-proteins', f'{root}/OGB',
                                     pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.node_species = None
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    data.y = data.y.float()

    evaluator = ogb.nodeproppred.Evaluator(name='ogbn-proteins')
    evaluator_wrapper = lambda pred, labels: evaluator.eval({"y_pred": pred, "y_true": labels})

    return PygNodeDataWrapper(data), dataset.num_features, data.y.shape[1], evaluator_wrapper, "rocauc"


def get_arxiv_pyg(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-arxiv', f'{root}/OGB',
                                     pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data.node_year = None
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)

    evaluator = ogb.nodeproppred.Evaluator(name='ogbn-arxiv')
    evaluator_wrapper = lambda pred, labels: evaluator.eval({"y_pred": pred[:,None], "y_true": labels[:,None]})

    return PygNodeDataWrapper(data), dataset.num_features, dataset.num_classes, evaluator_wrapper, "acc"


def get_citation2_pyg(root: str) -> Tuple[Data, int, int]:
    dataset = PygLinkPropPredDataset('ogbl-citation2', f'{root}/OGB',
                                     pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.x = None
    data.adj_t = data.adj_t.to_symmetric()
    data.node_year = None

    split_edge = dataset.get_edge_split()
    idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
    split_edge['eval_train'] = {
        'source_node': split_edge['train']['source_node'][idx],
        'target_node': split_edge['train']['target_node'][idx],
        'target_node_neg': split_edge['valid']['target_node_neg'],
    }

    evaluator = ogb.linkproppred.Evaluator(name='ogbl-citation2')
    evaluator_wrapper = lambda pred, labels: {'mrr_list': evaluator.eval({'y_pred_pos': pred[labels], 'y_pred_neg': pred[~labels]})['mrr_list'].mean().item()}

    return PygLinkDataWrapper(data, split_edge), dataset.num_features, dataset.num_classes, evaluator_wrapper, 'mrr_list'

def get_ppa_pyg(root: str) -> Tuple[Data, int, int]:
    dataset = PygLinkPropPredDataset('ogbl-ppa', f'{root}/OGB',
                                     pre_transform=T.ToSparseTensor())
    data = dataset[0]
    # data.x = None
    split_edge = dataset.get_edge_split()
    # import ipdb; ipdb.set_trace()
    # idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
    # split_edge['eval_train'] = {
    #     'edge': split_edge['train']['edge'][idx],
    #     'target_node': split_edge['train']['target_node'][idx],
    #     'edge_neg': split_edge['valid']['target_node_neg'],
    # }

    evaluator = ogb.linkproppred.Evaluator(name='ogbl-ppa')
    evaluator_wrapper = lambda pred, labels: evaluator.eval({'y_pred_pos': pred[labels], 'y_pred_neg': pred[~labels]})
    return PygLinkDataWrapper(data, split_edge), dataset.num_features, dataset.num_classes, evaluator_wrapper, 'hits@100'


def get_ppa_dgl(root: str):
    dataset = DglLinkPropPredDataset('ogbl-ppa', f'{root}/OGB',)
    data = dataset[0]
    # data.x = None
    split_edge = dataset.get_edge_split()
    evaluator = ogb.linkproppred.Evaluator(name='ogbl-ppa')
    # evaluator_wrapper = lambda pred, labels: evaluator.eval({'y_pred_pos': pred[labels], 'y_pred_neg': pred[~labels]})
    def evaluator_wrapper(pred, labels):
        if isinstance(labels, torch.Tensor):
            labels = labels.bool()
        else:
            labels = labels.astype(bool)
        results = {}
        for K in [20, 50, 100]:
            evaluator.K = K
            results.update(evaluator.eval({'y_pred_pos': pred[labels], 'y_pred_neg': pred[~labels]}))
        return results
    return DglLinkDataWrapper(data, split_edge), data.ndata["feat"].shape[-1], 0, evaluator_wrapper, 'hits@20'


def get_products_pyg(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-products', f'{root}/OGB',
                                     pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)

    evaluator = ogb.nodeproppred.Evaluator(name='ogbn-products')
    evaluator_wrapper = lambda pred, labels: evaluator.eval({"y_pred": pred[:,None], "y_true": labels[:,None]})

    return PygNodeDataWrapper(data), dataset.num_features, dataset.num_classes, evaluator_wrapper, "acc"


def get_data(root: str, name: str, mode='pyg') -> Tuple[Data, int, int]:
    if mode in ['pyg']:
        if name.lower() in ['ogbn-arxiv', 'arxiv']:
            return get_arxiv_pyg(root)
        elif name.lower() in ['ogbn-proteins', 'proteins']:
            return get_proteins_pyg(root)
        elif name.lower() in ['ogbl-ppa', 'ppa']:
            return get_ppa_pyg(root)
        elif name.lower() in ['ogbn-products', 'products']:
            return get_products_pyg(root)
        elif name.lower() in ['ogbl-citation2', 'citation2']:
            return get_citation2_pyg(root)
        else:
            raise NotImplementedError
    elif mode in ['dgl']:
        if name.lower() in ['ogbn-proteins', 'proteins']:
            return get_proteins_dgl(root)
        elif name.lower() in ['ogbn-products', 'products']:
            return get_products_dgl(root)
        elif name.lower() in ['ogbl-ppa', 'ppa']:
            return get_ppa_dgl(root)
        elif name.lower() in ['ogbn-arxiv', 'arxiv']:
            return get_arxiv_dgl(root)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
