
from torch_sparse import fill_diag, mul
from torch_sparse import sum as sparsesum


def udf_norm(edge_index, adj_type, add_self_loops=False, fill_value=1):
    adj_t = edge_index
    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1.,)
    if add_self_loops:
        adj_t = fill_diag(adj_t, fill_value)
    deg = sparsesum(adj_t, dim=1)
    if adj_type in ['DAD']:
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    elif adj_type in ['DA']:
        deg_inv_sqrt = deg.pow_(-1.0)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    elif adj_type in ['AD']:
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t