from typing import Any
import torch
from bert.bert_gnn_sampler import RecursiveSampler, OneshotSampler, TinySample
from bert.bert_utils import merge_src_dst

class GAMLPNodeData(torch.utils.data.Dataset):  # Map style
    def __init__(self, all_idx, labels, train_mask):
        super().__init__()
        self.all_idx = all_idx
        self.labels = labels
        self.train_mask = train_mask

    def get_batches(self, node_id):
        item = {}
        # item['gnn_n_id'] = self.all_idx[node_id]
        item['labels'] = self.labels[node_id]
        item['mask'] = self.train_mask[node_id]
        return self.all_idx[node_id], self.all_idx[node_id], item

    def __getitem__(self, node_id):
        result = self.get_batches(node_id)
        return result

    def __len__(self):
        return len(self.all_idx)
    
class ResampleWrapper():
    def __init__(self, data_loader) -> None:
        self.data_loader = data_loader

    def __iter__(self):
        self.data_loader.dataset.resample()
        sampler_iter = iter(self.data_loader)
        while True:
            try:
                result = next(sampler_iter)
                yield result
            except StopIteration:
                break

    def __len__(self) -> int:
        return len(self.data_loader)

def get_gamlp_node_loader(data, conf):
    all_idx = torch.arange(data.num_nodes)

    eval_dataset = GAMLPNodeData(all_idx, data.y, data.train_mask)

    eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=conf.model.params.batch_size, shuffle=False, drop_last=False)
    
    train_loader_func = lambda lm_batch_size: torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=lm_batch_size,
            # shuffle=True,
            drop_last=True,
            sampler=TinySample(data.train_mask, valid_test_dp=conf.model.params.valid_test_dp),
            # sampler=TinySample(data.train_mask, valid_test_dp=0.0),
            # num_workers=self.args.dataloader_num_workers,
            # pin_memory=self.args.dataloader_pin_memory,
        )

    
    return data, train_loader_func, eval_dataloader

    
class GAMLPLinkData(torch.utils.data.Dataset):
    # Base on SimpleTGData
    def __init__(self, all_idx, edge, edge_neg=None):
        super().__init__()
        self.src_id = edge[:,0]
        self.dst_id = edge[:,1]
        self.labels = torch.ones_like(edge[:,0])

        if edge_neg is not None:
            self.src_id = torch.cat([self.src_id, edge_neg[:,0]])
            self.dst_id = torch.cat([self.dst_id, edge_neg[:,1]])
            self.labels = torch.cat([self.labels, torch.zeros_like(edge_neg[:,0])])
        self.simple_data = all_idx

        assert len(self.src_id) == len(self.dst_id)

    def resample(self):
        return
    
    def __getitem__(self, edge_id):
        batch_src_id = self.src_id[edge_id]
        batch_dst_id = self.dst_id[edge_id]
        
        # gnn_n_id, gnn_src_batch_id, gnn_dst_batch_id = merge_src_dst(batch_src_id, batch_dst_id)
        # item = self.simple_data[gnn_n_id]
        item = {
            'batch_src_id': batch_src_id,
            'batch_dst_id': batch_dst_id,
            'labels': self.labels[edge_id],
        }
        return item

    def __len__(self):
        return len(self.src_id)
    


class GAMLPLinkCollate():
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
            'gnn_src_batch_id': gnn_src_batch_id,
            'gnn_dst_batch_id': gnn_dst_batch_id,
            'gnn_n_id': gnn_n_id,
            **batches,
        })

        return gnn_n_id, gnn_n_id, item
    


def get_gamlp_link_loader(data, simple_data, conf):
    all_idx = torch.arange(data.num_nodes)

    eval_dataset = GAMLPLinkData(all_idx, data.split_edge['valid']['edge'], data.split_edge['valid']['edge_neg'])
    eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, collate_fn=GAMLPLinkCollate(simple_data), batch_size=conf.model.params.batch_size, shuffle=False, drop_last=False)
    
    train_dataset = GAMLPLinkData(all_idx, data.split_edge['train']['edge'],)
    train_loader_func = lambda lm_batch_size: torch.utils.data.DataLoader(
            train_dataset,
            batch_size=lm_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=GAMLPLinkCollate(simple_data),
            # sampler=TinySample(data.train_mask, valid_test_dp=conf.model.params.valid_test_dp),
            # num_workers=self.args.dataloader_num_workers,
            # pin_memory=self.args.dataloader_pin_memory,
        )

    
    return data, train_loader_func, eval_dataloader


def get_gamlp_loader(data, simple_data, conf):
    if conf.dataset.task in ['node']:
        return get_gamlp_node_loader(data, conf)
    elif conf.dataset.task in ['link']:
        return get_gamlp_link_loader(data, simple_data, conf)
    else:
        raise NotImplementedError