import logging
import time
import hydra
import yaml
from omegaconf import OmegaConf

import os
import torch
import torch.nn.functional as F

from transformers import AutoModel, TrainingArguments
from bert.bert_utils import get_token, SimpleTGData, calc_bsz_grad_acc, build_compute_metrics
from bert.bert_trainer import GLBaseTrainer
from data import get_data

import logger

from utils.function.os_utils import init_random_state

import argparse
import numpy as np
import pandas as pd
from transformers import TrainerCallback

try:
    seed = int(os.environ.get("PYTHON_RANDOM_SEED"))
    init_random_state(seed)
    seed_name = '_seed{}'.format(seed)
except:
    init_random_state(345)
    seed_name = ''

def split_simple_data(conf, simple_data, data,):
    if conf.dataset.task in ['node']:
        subset_data = lambda sub_idx: torch.utils.data.Subset(simple_data, sub_idx)
        datasets = {_: subset_data(torch.arange(data.num_nodes)[getattr(data, f'{_}_mask')])
                         for _ in ['train', 'val', 'test']}
        return datasets['train'], datasets['val'],  datasets['test'], None,
    else:
        raise NotImplementedError

def get_load_save_dir(conf):
    """
    Returns:
        load_dir:
            hidden_state = torch.load(os.path.join(load_dir, 'hidden_state.pt'))
            # HINT: os.path.join(load_dir, 'lm_gnn.ckpt') may not exist.
            state_dict = torch.load(os.path.join(load_dir, 'lm_gnn.ckpt'))
        save_dir:
            torch.save(hidden_state, os.path.join(save_dir, 'hidden_state.pt'))
            torch.save(model.state_dict(), os.path.join(save_dir, 'lm_gnn.ckpt'))
    """


    if conf.phase.params.finetune_prefix is None:

        load_dir = os.path.join(conf.LM.path, conf.dataset.name + seed_name)
        save_dir = os.path.join(os.path.expanduser(conf.phase.params.ckpt), conf.LM.name, conf.dataset.name + seed_name, conf.model.name, conf.phase.name)
    else:
        if conf.phase.name == 'pre_lm':
            load_dir = os.path.join(os.path.expanduser(conf.phase.params.ckpt), conf.LM.name, conf.dataset.name + seed_name, conf.model.name, conf.phase.params.finetune_prefix)
            save_dir = os.path.join(load_dir, conf.phase.name+f'_{conf.LM.params.architecture.use_log}_{conf.LM.params.architecture.label_smoothing_factor}_{conf.LM.params.load_best_model_at_end}_{conf.LM.params.architecture.pseudo_temp}_{conf.model.params.valid_test_dp}')

        elif conf.phase.name == 'pre_gnn':
            load_dir = os.path.join(os.path.expanduser(conf.phase.params.ckpt), conf.LM.name, conf.dataset.name + seed_name, conf.model.name, conf.phase.params.finetune_prefix+f'_{conf.LM.params.architecture.use_log}_{conf.LM.params.architecture.label_smoothing_factor}_{conf.LM.params.load_best_model_at_end}_{conf.LM.params.architecture.pseudo_temp}_{conf.model.params.valid_test_dp}')
            save_dir = os.path.join(load_dir, conf.phase.name)

        else:
            raise ValueError('Wrong phase name!')

    return load_dir, save_dir


def get_bz_and_gas(conf):
    # Pretrain on gold data
    batch_size, grad_acc_steps = calc_bsz_grad_acc(conf.LM.params.eq_batch_size, conf.LM.params.max_bsz)
    if conf.phase.name in ['pre_gnn']:
        if conf.phase.params.gnn_grad_acc:
            conf.model.params.batch_size = conf.model.params.batch_size // grad_acc_steps
        else:
            grad_acc_steps = 1
        logger.log(f'GNN Eq_batch_size = {conf.model.params.batch_size*grad_acc_steps}, bsz={conf.model.params.batch_size}, grad_acc_steps={grad_acc_steps}')
    return batch_size, grad_acc_steps

@hydra.main(config_path='conf', config_name='config')
def main(conf):

    logging.info('start')

    """
    Step 1: Process args
    """
    conf.model.params = conf.model.params[conf.dataset.name]
    conf.phase.params = conf.phase.params[conf.dataset.name]
    if conf.LM.name in conf.phase.params:
        conf.phase.params = conf.phase.params[conf.LM.name]
    if conf.model.name in conf.phase.params:
        conf.phase.params = conf.phase.params[conf.model.name]
    phase_params = conf.phase.params
    conf.LM.path = os.path.expanduser(conf.LM.path)
    logger.configure(dir='./log_dir/', format_strs=['stdout','log','csv',])
    dict_conf = yaml.safe_load(OmegaConf.to_yaml(conf))
    logger.save_conf(dict_conf)


    """
    Step 2: Load data and metrics
    """
    t = time.perf_counter()
    logger.log('Loading data...')
    data, in_channels, out_channels, evaluator_wrapper, metric_for_best_model = get_data(conf.root, conf.dataset.name, conf.model.data_mode)
    conf.LM.params = conf.LM.params[conf.dataset.name]

    # new code start
    ndata = get_token(conf.LM.params.token_folder, data.num_nodes, max_length=conf.LM.params.max_length)
    simple_data = SimpleTGData(data, ndata)

    compute_metrics = build_compute_metrics(evaluator_wrapper, metric_for_best_model)
    logger.log(f'Done! [{time.perf_counter() - t:.2f}s]')


    """
    Step 3: Load node encoder (transformer)
    """
    bert_model = AutoModel.from_pretrained(conf.LM.path)


    """
    Step 4:
        1. Load GNN and corresponding graph dataloader
        2. Process data and simple_data
    """
    perm = torch.arange(data.num_nodes)
    if conf.model.framework in ['gat']:
        from gnn.gat.loader import get_gat_loader
        from gnn.gat.model import get_model, GATBertNodeClassifier
        data, train_loader_func, eval_loader = get_gat_loader(data, conf)
        gnn_model = get_model(conf, bert_model, out_channels)
        bert_gnn_model = eval(conf.model.params.bert_gnn_model)
    elif conf.model.framework in ['gamlp']:
        from gnn.gamlp.loader import get_gamlp_loader
        from gnn.gamlp.model import get_model, GAMLPBertNodeClassifier
        data, train_loader_func, eval_loader = get_gamlp_loader(data, simple_data, conf)
        gnn_model = get_model(conf, bert_model, out_channels)
        bert_gnn_model = eval(conf.model.params.bert_gnn_model)
    elif conf.model.framework in ['revgat']:
        from gnn.revgat.loader import get_revgat_loader
        from gnn.revgat.model import get_model, REVGATBertNodeClassifier
        data, train_loader_func, eval_loader = get_revgat_loader(data, conf)
        gnn_model = get_model(conf, bert_model, out_channels)
        bert_gnn_model = eval(conf.model.params.bert_gnn_model)
    elif conf.model.framework in ['sagn']:
        from gnn.sagn.loader import get_sagn_loader
        from gnn.sagn.model import get_model, SAGNBertNodeClassifier
        data, train_loader_func, eval_loader = get_sagn_loader(data, conf)
        gnn_model = get_model(conf, bert_model, out_channels)
        bert_gnn_model = eval(conf.model.params.bert_gnn_model)
    else:
        raise NotImplementedError


    """
    Step 5: Build Bert+GNN model
    """
    batch_size, grad_acc_steps = get_bz_and_gas(conf)
    training_args = TrainingArguments(
            output_dir=phase_params.out_dir,
            evaluation_strategy='steps',
            eval_steps=phase_params.eval_steps,
            save_strategy='steps',
            save_steps=phase_params.eval_steps,
            learning_rate=phase_params.lr, weight_decay=phase_params.weight_decay,
            load_best_model_at_end=(conf.LM.params.load_best_model_at_end == 'T'), gradient_accumulation_steps=grad_acc_steps,
            save_total_limit=2,
            report_to='tensorboard',
            per_device_train_batch_size=batch_size, # unused
            per_device_eval_batch_size=batch_size * 6 if conf.LM.name in {'esm2_t12_35M_UR50D', 'esm2_t6_8M_UR50D'} else batch_size * 10,
            warmup_steps=phase_params.warmup_ratio,
            disable_tqdm=False,
            dataloader_drop_last=True,
            num_train_epochs=phase_params.epochs,
            dataloader_num_workers=1,
            metric_for_best_model='val_accuracy',
            label_names=['labels'],
            remove_unused_columns=False,
            fp16=True,  # if cf.hf_model=='microsoft/deberta-large' else False
            bf16_full_eval=conf.LM.params.bf16_full_eval,
            # logging_steps=10,
    )
    model = bert_gnn_model(
        data, out_channels,
        bert_model, gnn_model,
        feat_shrink=conf.LM.params.feat_shrink,
        **conf.LM.params.architecture,
    )
    
    """
    Step 6: Load the pretrained model if it exists
    """
    load_dir, save_dir = get_load_save_dir(conf)
    if os.path.exists(os.path.join(load_dir, 'lm_gnn.ckpt')):
        logger.log(f'Load model from {load_dir}')
        state_dict = torch.load(os.path.join(load_dir, 'lm_gnn.ckpt'), map_location='cpu')
        pretrain_dict = model.state_dict()

        for k, v in pretrain_dict.items():
            if k in state_dict:
                if conf.model.framework in ['sagn']:
                    if str(k)[:3] == 'gnn':
                        if pretrain_dict[k].shape != state_dict[k].shape:
                            pretrain_dict[k] = state_dict[k].squeeze(0)
                        else:
                            pretrain_dict[k] = state_dict[k]
                else:
                    if pretrain_dict[k].shape != state_dict[k].shape:
                        pretrain_dict[k] = state_dict[k].squeeze(0)
                    else:
                        pretrain_dict[k] = state_dict[k]
        model.load_state_dict(pretrain_dict)

    if conf.LM.name == 'distilbert-base-uncased':
        model.config.dropout = conf.LM.params.dropout
        model.config.attention_dropout = conf.LM.params.att_dropout
    elif hasattr(conf.LM.params, 'dropout') and hasattr(conf.LM.params, 'att_dropout'):
        logger.log('default dropout and attention_dropout are:', model.config.hidden_dropout_prob, model.config.attention_probs_dropout_prob)
        model.config.hidden_dropout_prob = conf.LM.params.dropout
        model.config.attention_probs_dropout_prob = conf.LM.params.att_dropout

    """
    Step 7: Build trainer
    """
    train_data, valid_data, test_data, data_collator = split_simple_data(conf, simple_data, data)
    if conf.dataset.task in ['node']:
        trainer_func = GLBaseTrainer
    else:
        raise NotImplementedError

    optimizers = (None, None)
    if conf.model.framework in ['revgat'] and conf.phase.name == 'pre_gnn':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=phase_params.lr, weight_decay=phase_params.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: epoch / 50 if epoch <= 50 else 1)
        optimizers = (optimizer, scheduler)

    trainer = trainer_func(
        gnn_train_loader_func = train_loader_func, gnn_eval_loader = eval_loader,
        semi_supervised_info={
            'labels': data.y,
            'train_mask': data.train_mask,
            'val_mask': data.val_mask,
            'test_mask': data.test_mask,
            'graph': data.data,
            'pre_gnn_bz': conf.model.params.batch_size,
            'admm_gnn_bz': conf.model.params.admm_batch_size if hasattr(conf.model.params, 'admm_batch_size') else batch_size,
            'valid_test_dp': conf.model.params.valid_test_dp,
            'whole_data': simple_data,
        },
        decay_scale=phase_params.decay_scale,
        lr_scale=phase_params.lr_scale,
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=optimizers,
    )

    """
    Step 8: Load hidden embeddings
        if the hidden embeddings do not exist, then we use trainer to inference the hidden embeddings
    """
    save_gnn_path=None
    if conf.phase.name == 'pre_gnn' and conf.phase.params.finetune_prefix is None:
        save_gnn_path = save_dir
        os.makedirs(save_gnn_path, exist_ok=True)
    elif conf.phase.name == 'pre_lm':
        save_gnn_path = save_dir + '/pre_gnn'
        os.makedirs(save_gnn_path, exist_ok=True)
    trainer.load_hidden_state(load_dir, perm, save_gnn_path)

    """
    Step 9: Train
        if conf.phase.name == 'pre_lm':
            train only bert_model
        elif conf.phase.name == 'pre_gnn':
            train only gnn_model
        elif conf.phase.name == 'admm':
            train both bert_model and gnn_model
    """
    start_time = time.time()
    trainer.train(conf.phase.name,)
    end_time = time.time()
    for log_history in trainer.state.log_history:
        if 'eval_val_accuracy' in log_history:
            logger.logkvs(log_history)
            logger.dumpkvs()

    """
    Step 10: Save the best model
    """
    if conf.phase.params.save_model:
        # ! Save BertClassifer Save model parameters
        os.makedirs(save_dir, exist_ok=True)
        torch.save(trainer.model.state_dict(), os.path.join(save_dir, 'lm_gnn.ckpt'))
        if conf.phase.name in ['pre_lm',]:
            trainer.phase = 'pre_lm'
            metrics = trainer.predict(test_data).metrics
            logger.log(metrics)
        logger.log('Start to save hidden state')
        trainer.load_hidden_state(save_dir, perm, save_gnn_path)
        logger.log(f'{conf.phase.name}: LM-GNN saved to {save_dir}')
        logger.log(f'{conf.phase.name}: cost time {end_time-start_time}')
    else:
        torch.save(trainer.model.state_dict(), os.path.join('./lm_gnn.ckpt'))

    logging.info('finish')


if __name__ == "__main__":
    main()