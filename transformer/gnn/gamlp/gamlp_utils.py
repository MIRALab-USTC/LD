import numpy as np
import torch
import torch.nn.functional as F

from torch_sparse import SparseTensor
import dgl.function as fn
import dgl

from tqdm import tqdm

def dgl_neighbor_average_labels(feat, g, label_num_hops):
    feat = feat.to(torch.float)
    print("Compute neighbor-averaged labels")
    with g.local_scope():
        for hop in tqdm(range(label_num_hops)):
            """
            Compute multi-hop neighbor-averaged node features
            """
            g.ndata["f"] = feat
            g.update_all(fn.copy_u("f", "msg"),
                        fn.mean("msg", "f"))
            feat = g.ndata.pop('f')
    return feat

def dgl_neighbor_average_features(feat, g, num_hops):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    with g.local_scope():
        g.ndata["feat_0"] = feat
        for hop in tqdm(range(1, num_hops)):
            g.update_all(fn.copy_u(f"feat_{hop-1}", "msg"),
                        fn.mean("msg", f"feat_{hop}"))
        res = []
        for hop in range(1, num_hops):
            res.append(g.ndata.pop(f"feat_{hop}"))
    return res

def prepare_label_emb(labels, n_classes, train_idx, valid_idx, test_idx, label_teacher_emb=None):
    if label_teacher_emb == None:
        y = np.zeros(shape=(labels.shape[0], int(n_classes)))
        y[train_idx] = F.one_hot(labels[train_idx].to(
            torch.long), num_classes=n_classes).float().squeeze(1)
        y = torch.Tensor(y)
    else:
        print("use teacher label")
        y = np.zeros(shape=(labels.shape[0], int(n_classes)))
        y[valid_idx] = label_teacher_emb[len(
            train_idx):len(train_idx)+len(valid_idx)]
        y[test_idx] = label_teacher_emb[len(
            train_idx)+len(valid_idx):len(train_idx)+len(valid_idx)+len(test_idx)]
        y[train_idx] = F.one_hot(labels[train_idx].to(
            torch.long), num_classes=n_classes).float().squeeze(1)
        y = torch.Tensor(y)

    return y

def neighbor_average_labels(labels, adj, label_num_hops):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged labels")
    for hop in tqdm(range(label_num_hops)):
        labels = adj @ labels
    return labels



def neighbor_average_features(feat, adj, num_hops):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    res = []
    for hop in tqdm(range(num_hops-1)):
        feat = adj @ feat
        res.append(feat.clone())
    return res

def get_diag(adj_t):
    row, col, val = adj_t.coo()
    return val[row == col]

def loop_importance_simple(adj, num_hops):
    print("Compute loop-importance")
    res = []
    adj_diag = get_diag(adj)
    feat = adj_diag.clone()
    res.append(feat[:,None].clone())
    for hop in range(num_hops-1):
        feat = adj_diag * feat
        res.append(feat[:,None].clone())
    return res