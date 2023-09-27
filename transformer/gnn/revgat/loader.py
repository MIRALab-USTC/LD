import time
from utils.pyg_utils import udf_norm
from bert.bert_gnn_sampler import RecursiveSampler, OneshotSampler
import torch
import dgl


def preprocess(graph):
    global n_node_feats

    # make bidirected
    feat = graph.ndata["feat"]
    labels = graph.ndata["labels"]
    train_mask = graph.ndata["train_mask"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat
    graph.ndata["labels"] = labels
    graph.ndata["train_mask"] = train_mask

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph


def get_revgat_loader(data, conf):
    graph = preprocess(data.data)
    eval_loader = [[torch.tensor([0]), torch.tensor([0]), graph.cpu()]]
    #输入全图
    all_idx = torch.arange(data.data.num_nodes())
    n_id_in = all_idx[data.data.ndata['train_mask']]
    n_id_out = all_idx[data.data.ndata['train_mask']]
    train_loader_func = lambda gnn_batch_size: [n_id_in, n_id_out, graph.cpu()]

    return data, train_loader_func, eval_loader