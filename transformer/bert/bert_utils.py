# new code start
import os
import torch
import numpy as np
from types import SimpleNamespace as SN
import subprocess as sp
from pathlib import Path

import torch.nn.functional as F


def merge_src_dst(lm_n_id_src, lm_n_id_dst):
    '''
    Returen:
        lm_n_id, lm_src_batch_id, lm_dst_batch_id
    Such that:
        lm_n_id_src = lm_n_id[lm_src_batch_id]
        lm_n_id_dst = lm_n_id[lm_dst_batch_id]

        lm_n_id = torch.cat([lm_n_id_src, lm_n_id_dst])[]
    '''
    batch_size = lm_n_id_src.shape[0]
    assert lm_n_id_src.shape[0] == lm_n_id_dst.shape[0]

    lm_n_id, indice = torch.cat([lm_n_id_src, lm_n_id_dst]).unique(return_inverse=True)
    lm_src_batch_id, lm_dst_batch_id = indice.split(batch_size)
    return lm_n_id, lm_src_batch_id, lm_dst_batch_id


def aug_compute_loss(logits, labels, loss_func, is_gold=None, pl_weight=0.5,):
    deal_nan = lambda x: 0 if torch.isnan(x) else x
    mle_loss = deal_nan(loss_func(logits[is_gold], labels[is_gold]))
    pl_loss = deal_nan(loss_func(logits[~is_gold], labels[~is_gold]))
    loss = pl_weight * pl_loss + (1 - pl_weight) * mle_loss
    return loss

def get_token(token_folder, num_nodes, perm=None, max_length=512):
    assert os.path.exists(token_folder)
    ndata = {}
    info = {
    'input_ids': SN(shape=(num_nodes, max_length), type=np.uint16),
    'attention_mask': SN(shape=(num_nodes, max_length), type=bool),
    'token_type_ids': SN(shape=(num_nodes, max_length), type=bool)
    }
    for k in ['attention_mask', 'input_ids', 'token_type_ids']:
        i = info[k]
        if os.path.exists(f'{token_folder}/{k}.npy'):
            if perm is not None:
                ndata[k] = np.memmap(f'{token_folder}/{k}.npy', mode='r', dtype=i.type, shape=i.shape)[perm]
            else:
                ndata[k] = np.memmap(f'{token_folder}/{k}.npy', mode='r', dtype=i.type, shape=i.shape)
    return ndata



class TinyData(torch.utils.data.Dataset):
    def __init__(self, data, train_mask, valid_test_dp=1.0,):
        self.data = data
        self.valid_test_dp = valid_test_dp

        self.train_id = torch.arange(len(train_mask))[train_mask]
        self.valid_test_id = torch.arange(len(train_mask))[~train_mask]
        self.train_mask = train_mask

        self.sample_num = int(valid_test_dp * len(self.valid_test_id))
        self.resample()
        
        super().__init__()

    def resample(self):
        sample_valid_test_id = self.valid_test_id[torch.randperm(len(self.valid_test_id))[:self.sample_num]]
        self.sample_id = torch.cat([self.train_id, sample_valid_test_id])

    def __getitem__(self, node_id):
        sample_id = self.sample_id[node_id]
        item = self.data.get_batches(sample_id)
        return item

    def __len__(self):
        return len(self.train_id) + self.sample_num
    

# class LinkTGWrapper(torch.utils.data.Dataset):
#     # Base on SimpleTGData
#     def __init__(self, simple_data, src_id, dst_id):
#         super().__init__()
#         self.simple_data = simple_data
#         self.src_id = src_id
#         self.dst_id = dst_id

#         assert len(self.src_id) == len(self.dst_id)

#     def resample(self):
#         return
    
#     def __getitem__(self, edge_id):
#         batch_src_id = self.src_id[edge_id]
#         batch_dst_id = self.dst_id[edge_id]
        
#         item_src = self.simple_data.get_batches(batch_src_id, prefix='src')
#         item_dst = self.simple_data.get_batches(batch_dst_id, prefix='dst')
#         return {
#             **item_src, **item_dst,
#         }

#     def __len__(self):
#         return len(self.src_id)


class LinkCollate():
    def __init__(self, simple_data,) -> None:
        self.simple_data = simple_data
        self.default_collate = torch.utils.data.dataloader.default_collate
    
    def __call__(self, batches):
        batches = self.default_collate(batches)
        batch_src_id = batches.pop('batch_src_id')
        batch_dst_id = batches.pop('batch_dst_id')

        gnn_n_id, gnn_src_batch_id, gnn_dst_batch_id = merge_src_dst(batch_src_id, batch_dst_id)
        item = self.simple_data[gnn_n_id]
        item.update({
            'lm_src_batch_id': gnn_src_batch_id,
            'lm_dst_batch_id': gnn_dst_batch_id,
            'lm_n_id': gnn_n_id,
            **batches,
        })

        return item
    
    
class LinkTGWrapper(torch.utils.data.Dataset):
    # Base on SimpleTGData
    def __init__(self, edge, edge_neg=None):
        super().__init__()
        self.src_id = edge[:,0]
        self.dst_id = edge[:,1]
        self.labels = torch.ones_like(edge[:,0])

        if edge_neg is not None:
            self.src_id = torch.cat([self.src_id, edge_neg[:,0]])
            self.dst_id = torch.cat([self.dst_id, edge_neg[:,1]])
            self.labels = torch.cat([self.labels, torch.zeros_like(edge_neg[:,0])])

        assert len(self.src_id) == len(self.dst_id)

    def resample(self):
        return
    
    def __getitem__(self, edge_id):
        batch_src_id = self.src_id[edge_id]
        batch_dst_id = self.dst_id[edge_id]
        item = {
            'batch_src_id': batch_src_id,
            'batch_dst_id': batch_dst_id,
            'labels': self.labels[edge_id],
        }
        return item

    def __len__(self):
        return len(self.src_id)

    


class SimpleTGData(torch.utils.data.Dataset):  # Map style
    '''
    Return:
        Dict{
            attention_mask
            input_ids/inputs_embeds
            token_type_ids
            labels
            mask
            lm_n_id
        }
    '''
    def __init__(self, data, ndata, n_labels=None, mode='eval'):
        super().__init__()
        # self.data = data
        self.num_nodes = data.num_nodes
        self.ndata = ndata
        self.ndata['lm_n_id'] = torch.arange(data.num_nodes).long()
        if data.train_mask is not None:
            self.ndata['mask'] = data.train_mask
        if data.y is not None:
            self.ndata['labels'] = data.y
        # self.ndata['pesudo_labels'] = F.one_hot(self.data.y, num_classes=n_labels).type(torch.FloatTensor)

    def resample(self):
        return

    def permute(self, perm):
        for key in self.ndata:
            self.ndata[key] = self.ndata[key][perm]
        self.ndata['lm_n_id'] = torch.arange(self.num_nodes).long()

    def get_batches(self, node_id, prefix=''):
        _load = lambda k: torch.IntTensor(np.array(self.ndata[k][node_id]))
        # item = {k: _load(k) for k in self.token_keys if k != 'input_ids'}
        item = {}
        if 'attention_mask' in self.ndata:
            item[prefix+'attention_mask'] = _load('attention_mask')
        if 'inputs_embeds' in self.ndata:
            item[prefix+'inputs_embeds'] = torch.Tensor(np.array(self.ndata['inputs_embeds'][node_id]))
        else:
            item[prefix+'input_ids'] = torch.IntTensor(np.array(self.ndata['input_ids'][node_id]).astype(np.int32))
        if 'token_type_ids' in self.ndata:
            item[prefix+'token_type_ids'] = _load('token_type_ids')
        
        if 'labels' in self.ndata:
            item[prefix+'labels'] = self.ndata['labels'][node_id]
        if 'mask' in self.ndata:
            item[prefix+'mask'] = self.ndata['mask'][node_id]
        # item['pesudo_labels'] = _load('pesudo_labels')
        item[prefix+'lm_n_id'] = self.ndata['lm_n_id'][node_id]
        return item

    def __getitem__(self, node_id):
        item = self.get_batches(node_id)
        return item

    def __len__(self):
        return self.num_nodes
    
class GATTGData(torch.utils.data.Dataset):  # Map style
    def __init__(self, data, ndata, n_labels, mode='eval'):
        super().__init__()
        self.data = data
        self.ndata = ndata
        self.ndata['lm_n_id'] = torch.arange(self.data.number_of_nodes()).long()

    def get_batches(self, node_id):
        item = {}
        item['input_ids'] = torch.IntTensor(np.array(self.ndata['input_ids'][node_id]).astype(np.int32))
        item['labels'] = self.data.ndata["labels"][node_id]
        # item['pesudo_labels'] = _load('pesudo_labels')
        item['lm_n_id'] = self.ndata['lm_n_id'][node_id]
        return item

    def __getitem__(self, node_id):
        item = self.get_batches(node_id)
        return item

    def __len__(self):
        return self.data.number_of_nodes()
    

# *  <<<<<<<<<<<<<<<<<<<< PROJ SHARED UTILS >>>>>>>>>>>>>>>>>>>>
def floor_quantize(val, to_values):
    """Quantize a value with regard to a set of allowed values.

    Examples:
        quantize(49.513, [0, 45, 90]) -> 45
        quantize(17, [0, 10, 20, 30]) -> 10 # FLOORED

    Note: function doesn't assume to_values to be sorted and
    iterates over all values (i.e. is rather slow).

    Args:
        val        The value to quantize
        to_values  The allowed values
    Returns:
        Closest value among allowed values.
    """
    best_match = None
    best_match_diff = None
    assert min(to_values) <= val
    for other_val in to_values:
        if other_val <= val:  # Floored (only smaller values are matched)
            diff = abs(other_val - val)
            if best_match is None or diff < best_match_diff:
                best_match = other_val
                best_match_diff = diff
    return best_match


def get_max_batch_size(gpu_mem, max_bsz_dict):
    quantized_gpu_mem = floor_quantize(gpu_mem, max_bsz_dict.keys())
    return max_bsz_dict[quantized_gpu_mem]


def calc_bsz_grad_acc(eq_batch_size, max_bsz_dict, min_bsz=2):
    sv_info = ServerInfo()
    max_bsz_per_gpu = get_max_batch_size(sv_info.gpu_mem, max_bsz_dict)
    gpus = os.environ['CUDA_VISIBLE_DEVICES']
    n_gpus = len(gpus.split(',')) if gpus != '' else 1
    print(f'N-GPUs={n_gpus}')

    def find_grad_acc_steps(bsz_per_gpu):
        # Find batch_size and grad_acc_steps combination that are DIVISIBLE!
        grad_acc_steps = eq_batch_size / bsz_per_gpu / n_gpus
        if grad_acc_steps.is_integer():
            return bsz_per_gpu, int(grad_acc_steps)
        elif grad_acc_steps:
            if bsz_per_gpu >= min_bsz:
                return find_grad_acc_steps(bsz_per_gpu - 1)
            else:
                raise ValueError(f'Cannot find grad_acc_step with integer batch_size greater than {min_bsz}, eq_bsz={eq_batch_size}, n_gpus={n_gpus}')

    batch_size, grad_acc_steps = find_grad_acc_steps(max_bsz_per_gpu)
    print(f'Eq_batch_size = {eq_batch_size}, bsz={batch_size}, grad_acc_steps={grad_acc_steps}, ngpus={n_gpus}')
    return batch_size, grad_acc_steps


LINUX_HOME = str(Path.home())


class ServerInfo:
    def __init__(self):
        self.gpu_mem, self.gpus, self.n_gpus = 0, [], 0
        try:
            command = "nvidia-smi --query-gpu=memory.total --format=csv"
            gpus = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
            self.gpus = np.array(range(len(gpus)))
            self.n_gpus = len(gpus)
            self.gpu_mem = round(int(gpus[0].split()[0]) / 1024)
            self.sv_type = f'{self.gpu_mem}Gx{self.n_gpus}'
        except:
            print('NVIDIA-GPU not found, set to CPU.')
            self.sv_type = f'CPU'

    def __str__(self):
        return f'SERVER INFO: {self.sv_type}'
    

from datasets import load_metric
METRICS = {  # metric -> metric_path
    'accuracy': 'utils/function/hf_accuracy.py',
}
metrics = {m: load_metric(m_path) for m, m_path in METRICS.items()}
def compute_metrics(pred):
    """
        pred包含predictions, label_ids, inputs
    """
    if len(pred.label_ids.shape) == 1:
        predictions, references = pred.predictions.argmax(1), pred.label_ids
    else:
        predictions, references = pred.predictions.argmax(1), pred.label_ids.argmax(1)
    result_metric = {}
    for m_name, metric in metrics.items():
        result_metric.update(metric.compute(predictions=predictions, references=references) if m_name in {'accuracy', 'pearsonr', 'spearmanr'} else metric.compute(predictions=predictions, references=references, average='macro'))
    return result_metric


def compute_metrics_all(predictions, label_ids, train_mask, val_mask, test_mask):
    """
        pred包含predictions, label_ids, inputs
    """
    if len(label_ids.shape) == 1:
        predictions, references = predictions.argmax(1), label_ids
    else:
        predictions, references = predictions.argmax(1), label_ids.argmax(1)
    result_metric = {}
    for prefix, mask in zip(['train', 'val', 'test'], [train_mask, val_mask, test_mask]):
        for m_name, metric in metrics.items():
            temp_result_metric = metric.compute(predictions=predictions[mask], references=references[mask]) if m_name in {'accuracy', 'pearsonr', 'spearmanr'} else metric.compute(predictions=predictions[mask], references=references[mask], average='macro')
            for key, val in temp_result_metric.items():
                result_metric[prefix+'_'+key] = val
    return result_metric

def build_compute_metrics(evaluator, metric_for_best_model):
    def compute_metrics_udf(pred, train_mask=None, val_mask=None, test_mask=None):
        """
            pred包含predictions, label_ids, inputs
        """
        if pred.predictions.shape[-1] != 1:
            if len(pred.label_ids.shape) == 1:
                predictions, references = pred.predictions.argmax(1), pred.label_ids
            else:
                predictions, references = pred.predictions, pred.label_ids
        else:
            predictions, references = pred.predictions[:,0], pred.label_ids
        result_metric = {}
        if train_mask is not None:
            for prefix, mask in zip(['train', 'val', 'test'], [train_mask, val_mask, test_mask]):
                # result_metric[prefix+'_'+'accuracy'] = 
                result = evaluator(predictions[mask], references[mask])
                for metric_name in result:
                    result_metric[prefix+'_'+metric_name] = result[metric_name]
                result_metric[prefix+'_'+'accuracy'] = result[metric_for_best_model]
        else:
            result = evaluator(predictions, references)
            for metric_name in result:
                result_metric[metric_name] = result[metric_name]
            result_metric['val'+'_'+'accuracy'] = result[metric_for_best_model]
        return result_metric
    return compute_metrics_udf