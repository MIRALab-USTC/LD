import pandas as pd
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from Bio import SeqIO
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_folder', type=str,)
args = parser.parse_args()

node_dataset = PygNodePropPredDataset(name = 'ogbn-proteins', root=args.dataset_folder, transform=T.ToSparseTensor())
print(node_dataset[0])

node_nodeidx2proteinid = pd.read_csv(os.path.expanduser(os.path.join(args.dataset_folder, 'ogbn_proteins/mapping/nodeidx2proteinid.csv.gz')))

handle = open('./data/protein.sequences.v11.5.fa', 'r')
protein_sequences = SeqIO.parse(handle, 'fasta')
print(next(iter(protein_sequences)))

print('start transform')
t1 = time.time()
protein_dict = SeqIO.to_dict(protein_sequences, )
print(time.time()-t1)

from tqdm import tqdm
proteinseq_list = []
for proteinid in tqdm(node_nodeidx2proteinid['protein id']):
    proteinseq_list.append(protein_dict[proteinid].seq)
node_nodeidx2proteinid['protein seq'] = proteinseq_list
node_nodeidx2proteinid.to_csv(os.path.expanduser(os.path.join(args.dataset_folder, 'ogbn_proteins/mapping/nodeidx2proteinid.csv.gz')), index=False)

handle.close()
