import time
import hydra
import yaml
from omegaconf import OmegaConf
import os

import esm
from tqdm import tqdm
import pandas as pd
from torch.optim import SGD, Adagrad, Adadelta, RMSprop, Adam, NAdam
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.loader import NeighborLoader

from torch_geometric_autoscale import (get_data, metis, permute,
                                       SubgraphLoader, EvalSubgraphLoader,
                                       models, compute_micro_f1, dropout)

import logger


@hydra.main(config_path='conf', config_name='config')
def main(conf):
    conf.model.params = conf.model.params[conf.dataset.name]
    params = conf.model.params
    logger.configure(dir='./log_dir/', format_strs=['stdout','log','csv','tensorboard'])
    dict_conf = yaml.safe_load(OmegaConf.to_yaml(conf))
    logger.save_conf(dict_conf)
    # try:
    #     edge_dropout = params.edge_dropout
    # except:  # noqa
    #     edge_dropout = 0.0
    # grad_norm = None if isinstance(params.grad_norm, str) else params.grad_norm

    t = time.perf_counter()
    print('Loading data...', end=' ', flush=True)
    data, in_channels, out_channels = get_data(conf.root, conf.dataset.name)

    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    conf.LM.params = conf.LM.params[conf.dataset.name]
    node_feature = pd.read_csv(conf.LM.params.seq_path)


    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=512)
    model = model.cuda()
    model.eval()

    pbar = tqdm(total=len(node_feature)) 
    for i, seq in enumerate(node_feature['protein seq']):
        if 'J' in seq:
            node_feature.loc[i, 'protein seq'] = node_feature['protein seq'][i].replace("J", "X")
        if '*' in seq:
            node_feature.loc[i, 'protein seq'] = node_feature['protein seq'][i].replace("*", "")
        pbar.update(1)
    pbar.close()

    batch_size = 1000
    token_results = []
    pbar = tqdm(total=(len(node_feature)-1) // batch_size + 1) 
    for i in range((len(node_feature)-1) // batch_size + 1):

        batch_labels, batch_strs, batch_tokens = batch_converter(node_feature[['node idx','protein seq']][i*batch_size:(i+1)*batch_size].to_numpy())
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        token_results.append(batch_tokens)
        pbar.update(1)
    pbar.close()

    tokens = torch.cat(token_results, dim=0)
    os.makedirs(os.path.join(conf.LM.params.token_folder,), exist_ok=True)
    torch.save(tokens, os.path.join(conf.LM.params.token_folder, 'token.pt', ))
    




if __name__ == "__main__":
    main()