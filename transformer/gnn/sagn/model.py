import torch

from .utils.layer import *
from .utils.layer import GroupMLP
from ..gamlp.gamlp_utils import dgl_neighbor_average_features, dgl_neighbor_average_labels, prepare_label_emb
from bert.bert_model import NodeClassifier
import copy

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data.to(self.shadow[name].device) + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


################################################################
# Enhanced model with a label model in SLE
class SLEModel(nn.Module):
    def __init__(self, base_model, label_model, reproduce_previous=True):
        super().__init__()
        self._reproduce_previous = reproduce_previous
        self.base_model = base_model
        self.label_model = label_model
        self.reset_parameters()


    def reset_parameters(self):
        if self._reproduce_previous:
            self.previous_reset_parameters()
        else:
            if self.base_model is not None:
                self.base_model.reset_parameters()
            if self.label_model is not None:
                self.label_model.reset_parameters()

    def previous_reset_parameters(self):
        # To ensure the reproducibility of results from
        # previous (before clean up) version, we reserve
        # the old order of initialization.
        gain = nn.init.calculate_gain("relu")
        if self.base_model is not None:
            if hasattr(self.base_model, "multihop_encoders"):
                for encoder in self.base_model.multihop_encoders:
                    encoder.reset_parameters()
            if hasattr(self.base_model, "res_fc"):
                nn.init.xavier_normal_(self.base_model.res_fc.weight, gain=gain)
            if hasattr(self.base_model, "hop_attn_l"):
                if self.base_model._weight_style == "attention":
                    if self.base_model._zero_inits:
                        nn.init.zeros_(self.base_model.hop_attn_l)
                        nn.init.zeros_(self.base_model.hop_attn_r)
                    else:
                        nn.init.xavier_normal_(self.base_model.hop_attn_l, gain=gain)
                        nn.init.xavier_normal_(self.base_model.hop_attn_r, gain=gain)
            if self.label_model is not None:
                self.label_model.reset_parameters()
            if hasattr(self.base_model, "pos_emb"):
                if self.base_model.pos_emb is not None:
                    nn.init.xavier_normal_(self.base_model.pos_emb, gain=gain)
            if hasattr(self.base_model, "post_encoder"):
                self.base_model.post_encoder.reset_parameters()
            if hasattr(self.base_model, "bn"):
                self.base_model.bn.reset_parameters()

        else:
            if self.label_model is not None:
                self.label_model.reset_parameters()

    def forward(self, feats, label_emb):
        out = 0
        if self.base_model is not None:
            out = self.base_model(feats)

        if self.label_model is not None:
            if label_emb is not None:
                label_out = self.label_model(label_emb).mean(1)
                if isinstance(out, tuple):
                    out = (out[0] + label_out, out[1])
                else:
                    out = out + label_out

        return out

    def forward_lin(self, feats):
        out = 0
        if self.base_model is not None:
            out = self.base_model.pure_lin(feats)

        return out

class SAGN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, num_hops, n_layers, num_heads, weight_style="attention", alpha=0.5, focal="first",
                 hop_norm="softmax", dropout=0.5, input_drop=0.0, attn_drop=0.0, negative_slope=0.2, zero_inits=False, position_emb=False, cf=None):
        super(SAGN, self).__init__()

        self.cf = cf
        self.n_layers = n_layers
        self.num_hops = num_hops

        self._num_heads = num_heads
        self._hidden = hidden
        self._out_feats = out_feats
        self._weight_style = weight_style
        self._alpha = alpha
        self._hop_norm = hop_norm
        self._zero_inits = zero_inits
        self._focal = focal
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_drop)
        # self.bn = nn.BatchNorm1d(hidden * num_heads)
        self.bn = MultiHeadBatchNorm(num_heads, hidden * num_heads)
        self.relu = nn.ReLU()
        self.input_drop = nn.Dropout(input_drop)
        self.multihop_encoders = nn.ModuleList([GroupMLP(in_feats, hidden, hidden, num_heads, n_layers, dropout) for i in range(num_hops)])
        self.res_fc = nn.Linear(in_feats, hidden * num_heads, bias=False)

        if weight_style == "attention":
            self.hop_attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden)))
            self.hop_attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden)))
            self.leaky_relu = nn.LeakyReLU(negative_slope)

        if position_emb:
            self.pos_emb = nn.Parameter(torch.FloatTensor(size=(num_hops, in_feats)))
        else:
            self.pos_emb = None

        self.post_encoder = GroupMLP(hidden, hidden, out_feats, num_heads, n_layers, dropout)
        # self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for encoder in self.multihop_encoders:
            encoder.reset_parameters()
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        if self._weight_style == "attention":
            if self._zero_inits:
                nn.init.zeros_(self.hop_attn_l)
                nn.init.zeros_(self.hop_attn_r)
            else:
                nn.init.xavier_normal_(self.hop_attn_l, gain=gain)
                nn.init.xavier_normal_(self.hop_attn_r, gain=gain)
        if self.pos_emb is not None:
            nn.init.xavier_normal_(self.pos_emb, gain=gain)
        self.post_encoder.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, feats):
        out = 0
        feats = [self.input_drop(feat) for feat in feats]
        if self.pos_emb is not None:
            feats = [f + self.pos_emb[[i]] for i, f in enumerate(feats)]
        hidden = []
        for i in range(len(feats)):
            hidden.append(self.multihop_encoders[i](feats[i]).view(-1, self._num_heads, self._hidden))

        a = None
        if self._weight_style == "attention":
            if self._focal == "first":
                focal_feat = hidden[0]
            if self._focal == "last":
                focal_feat = hidden[-1]
            if self._focal == "average":
                focal_feat = 0
                for h in hidden:
                    focal_feat += h
                focal_feat /= len(hidden)

            astack_l = [(h * self.hop_attn_l).sum(dim=-1).unsqueeze(-1) for h in hidden]
            a_r = (focal_feat * self.hop_attn_r).sum(dim=-1).unsqueeze(-1)
            astack = torch.stack([(a_l + a_r) for a_l in astack_l], dim=-1)
            if self._hop_norm == "softmax":
                a = self.leaky_relu(astack)
                a = F.softmax(a, dim=-1)
            if self._hop_norm == "sigmoid":
                a = torch.sigmoid(astack)
            if self._hop_norm == "tanh":
                a = torch.tanh(astack)
            a = self.attn_dropout(a)

            for i in range(a.shape[-1]):
                out += hidden[i] * a[:, :, :, i]

        if self._weight_style == "uniform":
            for h in hidden:
                out += h / len(hidden)

        if self._weight_style == "exponent":
            for k, h in enumerate(hidden):
                out += self._alpha ** k * h

        out += self.res_fc(feats[0]).view(-1, self._num_heads, self._hidden)
        out = out.flatten(1, -1)
        out = self.dropout(self.relu(self.bn(out)))
        out = out.view(-1, self._num_heads, self._hidden)
        out = self.post_encoder(out)
        out = out.mean(1)

        return out, a.mean(1) if a is not None else None

    def pure_lin(self, feats):
        out = 0
        feats = self.input_drop(feats)
        if self.pos_emb is not None:
            feats = [f + self.pos_emb[[i]] for i, f in enumerate(feats)]
        hidden = self.multihop_encoders[0](feats).view(-1, self._num_heads, self._hidden)

        out = hidden

        out += self.res_fc(feats[0]).view(-1, self._num_heads, self._hidden)
        out = out.flatten(1, -1)
        out = self.dropout(self.relu(self.bn(out)))
        out = out.view(-1, self._num_heads, self._hidden)
        out = self.post_encoder(out)
        out = out.mean(1)

        return out


class R_GAMLP(nn.Module):  # recursive GAMLP
    def __init__(self, nfeat, hidden, nclass, num_hops,
                 dropout, input_drop, att_dropout, alpha, n_layers_1, n_layers_2, act="relu", pre_process=False, residual=False, pre_dropout=False, bns=False):
        super(R_GAMLP, self).__init__()
        self.num_hops = num_hops
        self.prelu = nn.PReLU()
        if pre_process:
            self.lr_att = nn.Linear(hidden + hidden, 1)
            self.lr_output = FeedForwardNetII(
                hidden, hidden, nclass, n_layers_2, dropout, alpha, bns)
            self.process = nn.ModuleList(
                [FeedForwardNet(nfeat, hidden, hidden, 2, dropout, bns) for i in range(num_hops)])
        else:
            self.lr_att = nn.Linear(nfeat + nfeat, 1)
            self.lr_output = FeedForwardNetII(
                nfeat, hidden, nclass, n_layers_2, dropout, alpha, bns)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.pre_process = pre_process
        self.res_fc = nn.Linear(nfeat, hidden)
        self.residual = residual
        self.pre_dropout = pre_dropout
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        if self.pre_process:
            for layer in self.process:
                layer.reset_parameters()

    def forward(self, feature_list):
        num_node = feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = []
        if self.pre_process:
            for i in range(self.num_hops):
                input_list.append(self.process[i](feature_list[i]))
        else:
            input_list = feature_list
        attention_scores = []
        attention_scores.append(self.act(self.lr_att(
            torch.cat([input_list[0], input_list[0]], dim=1))))
        for i in range(1, self.num_hops):
            history_att = torch.cat(attention_scores[:i], dim=1)
            att = F.softmax(history_att, 1)
            history = torch.mul(input_list[0], self.att_drop(
                att[:, 0].view(num_node, 1)))
            for j in range(1, i):
                history = history + \
                          torch.mul(input_list[j], self.att_drop(
                              att[:, j].view(num_node, 1)))
            attention_scores.append(self.act(self.lr_att(
                torch.cat([history, input_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores = F.softmax(attention_scores, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            attention_scores[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + \
                      torch.mul(input_list[i], self.att_drop(
                          attention_scores[:, i].view(num_node, 1)))
        if self.residual:
            right_1 += self.res_fc(feature_list[0])
            right_1 = self.dropout(self.prelu(right_1))
        if self.pre_dropout:
            right_1 = self.dropout(right_1)
        right_1 = self.lr_output(right_1)
        return right_1


class R_GAMLP_RLU(nn.Module):  # recursive GAMLP
    def __init__(self, nfeat, hidden, nclass, num_hops,
                 dropout, input_drop, att_dropout, label_drop, alpha, n_layers_1, n_layers_2, n_layers_3, act, pre_process=False, residual=False, pre_dropout=False, bns=False):
        super(R_GAMLP_RLU, self).__init__()
        self.num_hops = num_hops
        self.pre_dropout = pre_dropout
        self.prelu = nn.PReLU()
        if pre_process:
            self.lr_att = nn.Linear(hidden + hidden, 1)
            self.lr_output = FeedForwardNetII(
                hidden, hidden, nclass, n_layers_2, dropout, alpha, bns)
            self.process = nn.ModuleList(
                [FeedForwardNet(nfeat, hidden, hidden, 2, dropout, bns) for i in range(num_hops)])
        else:
            self.lr_att = nn.Linear(nfeat + nfeat, 1)
            self.lr_output = FeedForwardNetII(
                nfeat, hidden, nclass, n_layers_2, dropout, alpha, bns)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.pre_process = pre_process
        self.res_fc = nn.Linear(nfeat, hidden)
        self.label_drop = nn.Dropout(label_drop)
        self.residual = residual
        self.label_fc = FeedForwardNet(
            nclass, hidden, nclass, n_layers_3, dropout)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        self.reset_parameters()
        self.sig = torch.nn.Sigmoid()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        if self.pre_process:
            for layer in self.process:
                layer.reset_parameters()

    def forward(self, feature_list, label_emb):
        num_node = feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = []
        if self.pre_process:
            for i in range(self.num_hops):
                input_list.append(self.process[i](feature_list[i]))
        else:
            input_list = feature_list
        attention_scores = []
        attention_scores.append(self.act(self.lr_att(
            torch.cat([input_list[0], input_list[0]], dim=1))))
        for i in range(1, self.num_hops):
            history_att = torch.cat(attention_scores[:i], dim=1)
            att = F.softmax(history_att, 1)
            history = torch.mul(input_list[0], self.att_drop(
                att[:, 0].view(num_node, 1)))
            for j in range(1, i):
                history = history + \
                          torch.mul(input_list[j], self.att_drop(
                              att[:, j].view(num_node, 1)))

            attention_scores.append(self.act(self.lr_att(
                torch.cat([history, input_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores = F.softmax(attention_scores, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            attention_scores[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + \
                      torch.mul(input_list[i], self.att_drop(
                          attention_scores[:, i].view(num_node, 1)))
        if self.residual:
            right_1 += self.res_fc(feature_list[0])
            right_1 = self.dropout(self.prelu(right_1))
        if self.pre_dropout:
            right_1 = self.dropout(right_1)
        right_1 = self.lr_output(right_1)
        right_1 += self.label_fc(self.label_drop(label_emb))
        return right_1


def get_model(conf, bert_model, out_channels):
    in_channels = int(conf.LM.params.feat_shrink) if conf.LM.params.feat_shrink else bert_model.config.hidden_size
    cf = conf.model.params.architecture
    num_hops = cf.num_hops + 1

    base_model = SAGN(in_channels, cf.n_hidden, out_channels, num_hops,
                      cf.mlp_layer, cf.num_heads,
                      weight_style=cf.weight_style,
                      dropout=cf.dropout,
                      input_drop=cf.input_drop,
                      attn_drop=cf.att_drop,
                      zero_inits=cf.zero_inits,
                      position_emb=cf.position_emb,
                      cf=cf,
                      # focal=cf.focal,
                      )
    label_model = GroupMLP(out_channels,
                           cf.n_hidden,
                           out_channels,
                           cf.num_heads,
                           cf.label_mlp_layer,
                           cf.label_drop,
                           residual=cf.label_residual, )
    model = SLEModel(base_model, label_model)
    return model


class SAGNBertNodeClassifier(NodeClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feats = []
        for hop in range(1, self.gnn_model.base_model.num_hops + 1):
            self.feats.append(self.hist_emb.emb.clone())
        self.label_input = None
        self.cf = self.gnn_model.base_model.cf
        self._device = torch.device('cpu')
        
        self.all_eval_loader = torch.utils.data.DataLoader(range(self.data.num_nodes), batch_size=self.cf.eval_batch_size,shuffle=False, drop_last=False)

        if self.cf.mean_teacher == True:
            print("use teacher-SAGN")
            self.teacher_model = copy.deepcopy(self.gnn_model)
            self.teacher_model = self.teacher_model.to(self._device)
            for param in self.teacher_model.parameters():
                param.detach_()

        if self.cf.ema == True:
            print("use ema")
            self.ema = EMA(self.gnn_model, self.cf.decay)
            self.ema.register()
        else:
            self.ema = None

        self.step = 0
        self.enhance_idx_cons = []
        self.enhance_loader_cons=[]

    def inverse_label(self, graph, ):
        # num_propagation = len(self.gnn_model.process) - 1
        num_propagation = self.gnn_model.base_model.n_layers

        xs = dgl_neighbor_average_features(self.label_emb, graph, num_propagation + 1)
        xs = [self.label_emb.clone(), ] + xs
        self.inv_label_emb = torch.stack(xs, dim=-1)

    def build_aggr_weight(self, ):
        # self.aggr_weight = nn.Parameter(torch.ones(len(self.gnn_model.process)) / (len(self.gnn_model.process)))
        self.aggr_weight = nn.Parameter(torch.ones(self.gnn_model.base_model.n_layers+1)/(self.gnn_model.base_model.n_layers+1))

    def preprocess_feat_label(self, semi_supervised_info, all_preds=None):

        self.feats = dgl_neighbor_average_features(self.hist_emb.emb, semi_supervised_info['graph'],
                                                   self.gnn_model.base_model.num_hops)
        
        all_labels = semi_supervised_info['labels']

        index = torch.arange(semi_supervised_info['graph'].number_of_nodes())
        train_nid = index[semi_supervised_info['train_mask']]
        val_nid = index[semi_supervised_info['val_mask']]
        test_nid = index[semi_supervised_info['test_mask']]

        if self.label_input is None:
            self.label_input = prepare_label_emb(all_labels, self.n_labels, train_nid, val_nid, test_nid, None)
        else:
            self.label_input = prepare_label_emb(all_labels, self.n_labels, train_nid, val_nid, test_nid, None) # self.label_emb = None (at first)
        self.label_input = dgl_neighbor_average_labels(self.label_input, semi_supervised_info['graph'], self.cf.label_num_hops)


    def _train_mean_teacher(self, train_batch, enhance_batch, batch_label_emb, batch_label_emb_cons, gold_label, mask):
        # ! train_sagn_mean_teacher
        self.gnn_model.train()

        batch_feats = train_batch
        batch_feats_cons = enhance_batch
        if self.label_emb is not None:
            batch_label_emb = batch_label_emb.to(torch.device('cuda'))
            if batch_label_emb_cons is not None:
                batch_label_emb_cons = batch_label_emb_cons.to(torch.device('cuda'))
        else:
            batch_label_emb = None
            batch_label_emb_cons = None

        out, _ = self.gnn_model(batch_feats, batch_label_emb)

        if batch_label_emb_cons is not None:
            out_s, _ = self.gnn_model(batch_feats_cons, batch_label_emb_cons)
            out_t, _ = self.teacher_model(batch_feats_cons, batch_label_emb_cons)

            from .utils.utils import consis_loss_mean_teacher
            mse, kl = consis_loss_mean_teacher(out_t, out_s, self.cf.tem, self.cf.lam)
            kl = kl * self.cf.kl_lam
        else:
            mse, kl = 0, 0

        L1 = self.loss_func(out[mask], gold_label[mask].to(torch.device('cuda')))

        if self.cf.kl == False:
            loss = L1 + mse
        else:
            loss = L1 + kl

        return out, loss

    def gnn_inference(self, n_id_in, n_id_out, gnn_batch, cls_token_emb=None, lm_batch_id=None):
        
        n_id_in = n_id_in.cpu()
        if cls_token_emb is None:
            cls_token_emb = self.hist_emb.pull(n_id_in)

        labels = gnn_batch['labels']
        mask = gnn_batch['mask']

        batch_feats = [self.feats[i][n_id_in].to(labels.device) for i in range(len(self.feats))]
        batch_feats = [cls_token_emb, ] + batch_feats

        if self.gnn_model.training:
            self.step += 1
            if self.step < self.cf.warm_step:
                logits, _ = self.gnn_model(batch_feats, self.label_input[n_id_in].to(labels.device))
                loss = self.loss_func(logits[mask], labels[mask])
            else:

                if self.ema != None:
                    self.ema.update()

                alpha = self.cf.ema_decay
                for mean_param, param in zip(self.teacher_model.parameters(), self.gnn_model.parameters()):
                    mean_param.data.mul_(alpha).add_(1 - alpha, param.data)

                if self.step % self.cf.gap_step == 0 or self.step == self.cf.warm_step:
                    from .utils.utils import gen_output_torch
                    preds = gen_output_torch(self.gnn_model, self.feats, self.hist_emb, self.all_eval_loader, self._device,
                                             self.label_input, self.ema)
                    predict_prob = preds.softmax(dim=1)
                    threshold = self.cf.top - (self.cf.top - self.cf.down) * self.step / 3000

                    self.enhance_idx_cons = torch.arange(self.data.num_nodes)[
                        (predict_prob.max(1)[0] > threshold) & ~self.data.train_mask]

                    self.enhance_loader_cons = torch.utils.data.DataLoader(self.enhance_idx_cons, batch_size=len(n_id_in),
                                                                   shuffle=False, drop_last=False)
                    self.enhance_loader_cons = iter(self.enhance_loader_cons)

                if len(self.enhance_idx_cons) > 0:
                    try:
                        enhance_idx = next(self.enhance_loader_cons)
                    except:
                        self.enhance_loader_cons = iter(torch.utils.data.DataLoader(self.enhance_idx_cons, batch_size=len(n_id_in),
                                                                          shuffle=False, drop_last=False))
                        enhance_idx = next(self.enhance_loader_cons)
                    enhance_batch = [self.feats[i][enhance_idx].to(labels.device) for i in range(len(self.feats))]
                    enhance_cls_token_emb = self.hist_emb.pull(enhance_idx)
                    enhance_batch = [enhance_cls_token_emb, ] + enhance_batch
                    batch_label_emb_cons = self.label_input[enhance_idx]
                else:
                    enhance_batch = None
                    batch_label_emb_cons = None

                gold_label = labels
                batch_label_emb = self.label_input[n_id_in]
                train_batch = batch_feats
                logits, loss = self._train_mean_teacher(train_batch, enhance_batch, batch_label_emb, batch_label_emb_cons, gold_label, mask)

        else:
            logits, _ = self.gnn_model(batch_feats, self.label_input[n_id_in].to(labels.device))
            loss = self.loss_func(logits[mask], labels[mask])

        return loss, logits, gnn_batch.pop('labels')