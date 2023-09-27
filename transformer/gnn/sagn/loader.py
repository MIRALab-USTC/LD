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


def get_sagn_loader(data, conf):
    all_idx = torch.arange(data.num_nodes)

    eval_dataset = GAMLPNodeData(all_idx, data.y, data.train_mask)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=conf.model.params.batch_size, shuffle=False, drop_last=False)

    train_loader_func = lambda lm_batch_size: torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=lm_batch_size,
        # shuffle=True,
        drop_last=True,
        # sampler=TinySample(data.train_mask, valid_test_dp=conf.model.params.valid_test_dp),
        sampler=TinySample(data.train_mask, valid_test_dp=0.0),
    )

    return data, train_loader_func, eval_dataloader
