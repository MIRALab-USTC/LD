from typing import Optional, Callable
from collections.abc import Mapping

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from utils.function.os_utils import init_random_state
import numpy as np
from bert.bert_utils import aug_compute_loss

from bert.history import History


class BaseClassifier(PreTrainedModel):
    def lm_inference(self, **inputs):
        outputs = self.bert_encoder(
            **inputs,
            output_hidden_states=True)
        emb = outputs['hidden_states'][-1]  # outputs[0]=last hidden state

        # mean pool
        len_text = inputs['attention_mask'].sum(1, keepdim=True)
        cls_token_emb = (emb*inputs['attention_mask'][:,:,None]).sum(1)
        cls_token_emb = cls_token_emb/len_text

        # Use CLS Emb as sentence emb.
        # cls_token_emb = emb.permute(1, 0, 2)[0]

        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        return cls_token_emb

    def gnn_inference(self, n_id_in, n_id_out, gnn_batch, cls_token_emb=None, lm_batch_id=None):
        raise NotImplementedError
    
    def update_emb(self, **inputs):
        labels = inputs.pop("labels", None)
        lm_n_id = inputs.pop("lm_n_id", None)
        mask = inputs.pop("mask", None)

        cls_token_emb = self.lm_inference(**inputs)
        cls_token_emb = cls_token_emb.type(self.tensor_dtype)
        if self.update_hist:
            self.hist_emb.push(cls_token_emb, lm_n_id.cpu())

    def preprocess_feat_label(self, *args, **kwargs):
        return


class NodeClassifier(BaseClassifier):
    def __init__(self, data, n_labels,  model, gnn_model, mask_gnn, use_log=0, label_inverse=False, label_smoothing_factor=0.0, ce_reduction='mean', pseudo_label_weight=0.5, pseudo_temp=1.0, label_as_feat=True, update_hist=True, cla_dropout=0.0, cla_bias=True, feat_shrink='', coef_augmented=0.0, dtype=th.float32):
        super().__init__(model.config)
        num_nodes = data.num_nodes
        if label_inverse:
            loss_func = th.nn.CrossEntropyLoss(label_smoothing=0, reduction=ce_reduction) if data.y.dim() == 1 else th.nn.BCEWithLogitsLoss()
        else:
            loss_func = th.nn.CrossEntropyLoss(label_smoothing=label_smoothing_factor, reduction=ce_reduction) if data.y.dim() == 1 else th.nn.BCEWithLogitsLoss()
        self.label_smoothing_factor = label_smoothing_factor
        self.bert_encoder, self.loss_func = model, loss_func
        self.dropout = nn.Dropout(cla_dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.gnn_model = gnn_model

        self.hist_emb = History(num_nodes, hidden_dim, device=None, dtype=dtype)
        self.mask_gnn = mask_gnn
        self.tensor_dtype = dtype

        if hasattr(self.gnn_model, 'forward_lin'):
            self.classifier = self.gnn_model.forward_lin
        else:
            print('---rebuild classifier----')
            self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)

        self.coef_augmented = coef_augmented
        self.pseudo_label_weight = pseudo_label_weight
        self.pseudo_temp = pseudo_temp
        self.label_as_feat = label_as_feat
        self.update_hist = update_hist
        
        if data.y.dim() == 1:
            self.label_emb = F.one_hot(data.y, num_classes=n_labels).type(th.FloatTensor)
        else:
            self.label_emb = data.y.clone().type(th.FloatTensor)
            # self.is_augmented

        self.label_emb[~data.train_mask] = 0
        self.label_inverse = label_inverse
        self.use_log = use_log
        self.inv_label_emb = None
        self.build_aggr_weight()
        self.is_augmented = False
        self.n_labels = n_labels

        self.all_logits = None
        self.record_aggr_weight = []
        self.y_dim = data.y.dim()
        self.data = data

    def build_aggr_weight(self,):
        raise NotImplementedError

    def forward(self, **inputs):
        if "lm_n_id" in inputs:
            # has lm data
            labels = inputs.pop("labels")
            if labels.shape[-1] == 1 and len(labels.shape)>1:
                labels = labels.squeeze(-1)
            lm_n_id = inputs.pop("lm_n_id")
            mask = inputs.pop("mask")
        else:
            lm_n_id = None

        if "gnn_input" in inputs:
            # has gnn data
            n_id_in, n_id_out, gnn_input = inputs.pop("n_id_in"), inputs.pop("n_id_out"), inputs.pop("gnn_input")
            lm_batch_id = inputs.pop("lm_batch_id")
        else:
            gnn_input = None

        if lm_n_id is not None:
            # lm_inference
            cls_token_emb = self.lm_inference(**inputs)
            cls_token_emb = cls_token_emb.type(self.tensor_dtype)
            if self.update_hist:
                self.hist_emb.push(cls_token_emb, lm_n_id.cpu())
            cls_token_emb = self.dropout(cls_token_emb)
        else:
            assert gnn_input is not None
            cls_token_emb = self.hist_emb.pull(n_id_in.cpu())

        if gnn_input is not None:
            # gnn_inference
            loss, logits, gnn_labels = self.gnn_inference(n_id_in, n_id_out, gnn_input, cls_token_emb, lm_batch_id)

        else:
            logits = self.classifier(cls_token_emb)
            if self.use_log:
                auto_label= (self.inv_label_emb[lm_n_id].to(logits.device) @ th.softmax(self.aggr_weight, dim=-1))
                self.record_aggr_weight.append(th.softmax(self.aggr_weight, dim=-1).tolist())
            else:
                auto_label= (self.inv_label_emb[lm_n_id].to(logits.device) @ self.aggr_weight)

            if self.y_dim == 1:
                auto_label = auto_label / auto_label.sum(-1, keepdim=True)
            else:
                pass

            goal = self.label_smoothing_factor * auto_label + (1-self.label_smoothing_factor) * self.label_emb[lm_n_id].to(logits.device)

            # loss = self.loss_func(logits, goal)
            loss = aug_compute_loss(logits, goal, self.loss_func, is_gold=mask, pl_weight=self.pseudo_label_weight  if self.is_augmented else 0.0,)
        if (isinstance(loss, float)) and self.training:
            import ipdb; ipdb.set_trace()
        return TokenClassifierOutput(loss=loss, logits=logits,)


class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, loss_func, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder, self.loss_func = model, loss_func
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)

    def forward(self, **input):
        # Extract outputs from the model
        labels = input.pop("labels")
        lm_n_id = input.pop("lm_n_id")
        mask = input.pop("mask")
        
        # import ipdb; ipdb.set_trace()
        outputs = self.bert_encoder(
            **input,
            output_hidden_states=True)
        emb = self.dropout(outputs['hidden_states'][-1])  # outputs[0]=last hidden state
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.classifier(cls_token_emb)

        if labels.shape[-1] == 1 and len(labels.shape)>1:
            labels = labels.squeeze(-1)
        # print(f'{sum(is_gold)} gold, {sum(~is_gold)} pseudo')
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits,)
    

class TestBertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, loss_func, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder, self.loss_func = model, loss_func
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)

    def forward(self, **input):
        # Extract outputs from the model
        labels = input.pop("labels")
        lm_n_id = input.pop("lm_n_id")
        
        self.bert_encoder.eval()
        outputs = self.bert_encoder(
            **input,
            output_hidden_states=True)
        emb = self.dropout(outputs['hidden_states'][-1])  # outputs[0]=last hidden state
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.classifier(cls_token_emb)

        if labels.shape[-1] == 1 and len(labels.shape)>1:
            labels = labels.squeeze(-1)
        # print(f'{sum(is_gold)} gold, {sum(~is_gold)} pseudo')
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits,)