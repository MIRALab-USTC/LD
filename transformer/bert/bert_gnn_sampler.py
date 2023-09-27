from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Sized, Iterator, NamedTuple
from collections.abc import Mapping
import torch
from torch.utils.data import DataLoader, Sampler
from torch_sparse import SparseTensor


# class GNNInput(NamedTuple):
#     adj_t: SparseTensor
#     batch_size: int
#     n_id: torch.Tensor  # The indices of mini-batched nodes
#     offset: torch.Tensor  # The offset of contiguous mini-batched nodes
#     count: torch.Tensor  # The number of contiguous mini-batched nodes
#     lmc_params: Dict

#     def to(self, *args, **kwargs):
#         return GNNInput(self.adj_t.to(*args, **kwargs), self.batch_size,
#                        self.n_id, self.offset, self.count, self.lmc_params)
class TinySample(Sampler[int]):
    def __init__(self, train_mask, valid_test_dp=1.0,):
        self.valid_test_dp = valid_test_dp

        self.train_id = torch.arange(len(train_mask))[train_mask]
        self.valid_test_id = torch.arange(len(train_mask))[~train_mask]
        self.train_mask = train_mask

        self.sample_num = int(valid_test_dp * len(self.valid_test_id))
        self.resample()

    def resample(self):
        sample_valid_test_id = self.valid_test_id[torch.randperm(len(self.valid_test_id))[:self.sample_num]]
        self.sample_id = torch.cat([self.train_id, sample_valid_test_id])

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        return len(self.train_id) + self.sample_num

    def __iter__(self) -> Iterator[int]:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        self.resample()
    
        yield from self.sample_id[torch.randperm(len(self.train_id) + self.sample_num, generator=generator)].tolist()
        
    def __len__(self) -> int:
        return self.num_samples



class RecursiveSampler(Sampler[int]): # drop last
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, gnnloader: DataLoader, lm_batch_size, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source # lm dataset, huggingface fashion
        self.gnnloader = gnnloader # gnn loader
        self.replacement = replacement
        self.generator = generator
        self.lm_batch_size = lm_batch_size
        

        self.gnn_data_list = list(self.gnnloader)
        self.lmindex_list = []
        for i, (n_id_in, n_id_out, gnn_input) in enumerate(self.gnn_data_list):
            index_dataset = DataLoader(torch.arange(len(n_id_out)), batch_size=self.lm_batch_size, shuffle=True, drop_last=True)
            self.lmindex_list += [(index, i) for index in index_dataset]

        assert self.replacement == False
        assert num_samples is None

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        return len(self.lmindex_list)

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        self.gnn_data_list = list(self.gnnloader)
        self.lmindex_list = []
        for i, (n_id_in, n_id_out, gnn_input) in enumerate(self.gnn_data_list): # data, batch_size, n_id, *args
            batch_size = len(n_id_out)
            index_dataset = DataLoader(torch.arange(batch_size), batch_size=self.lm_batch_size, shuffle=True, drop_last=True)
            self.lmindex_list += [(index, i) for index in index_dataset]
        
        index_shuffle = torch.randperm(len(self.lmindex_list), generator=generator)
        

        sampler_iter = iter(index_shuffle)
        while True:
            try:
                batch_index = next(sampler_iter)
                lm_batch_id, cluster_id = self.lmindex_list[batch_index]

                n_id_in, n_id_out, gnn_input = self.gnn_data_list[cluster_id]
                lm_n_id = n_id_out[lm_batch_id]
                lm_data = self.data_source[lm_n_id]
                result =  {
                    'n_id_in': n_id_in,
                    'n_id_out': n_id_out,
                    'gnn_input': gnn_input,
                    'lm_batch_id': lm_batch_id,
                }
                lm_data.update(result)
                yield lm_data
            except StopIteration:
                break

    def __len__(self) -> int:
        return self.num_samples


class OneshotSampler(Sampler[int]): # drop last
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, gnnloader: DataLoader, lm_batch_size = None, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source # lm data
        self.gnnloader = gnnloader # gnn loader
        self.replacement = replacement
        self.generator = generator
        self.lm_batch_size = lm_batch_size # unused
        

        

        assert self.replacement == False
        assert num_samples is None

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        return len(self.gnnloader)

    def __iter__(self) -> Iterator[int]:
        
        sampler_iter = iter(self.gnnloader)
        while True:
            try:
                n_id_in, n_id_out, gnn_input = next(sampler_iter)
                assert torch.all(n_id_in[:len(n_id_out)] == n_id_out)
                lm_batch_id = torch.arange(len(n_id_out))
                lm_n_id = n_id_out
                lm_data = self.data_source[lm_n_id]
                result =  {
                    'n_id_in': n_id_in,
                    'n_id_out': n_id_out,
                    'gnn_input': gnn_input,
                    'lm_batch_id': lm_batch_id,
                }
                lm_data.update(result)
                if 'gnn_src_batch_id' in gnn_input:
                    lm_data['lm_src_batch_id'] = gnn_input['gnn_src_batch_id']
                    lm_data['lm_dst_batch_id'] = gnn_input['gnn_dst_batch_id']
                yield lm_data
            except StopIteration:
                break

    def __len__(self) -> int:
        return self.num_samples



class DictWrapper(Sampler[int]):
    data_source: Sized
    replacement: bool

    def __init__(self, gnnloader: DataLoader,) -> None:
        self.gnnloader = gnnloader # gnn loader

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        return len(self.gnnloader)

    def __iter__(self) -> Iterator[int]:
        
        sampler_iter = iter(self.gnnloader)
        while True:
            try:
                if isinstance(self.gnnloader, list):
                    n_id_in, n_id_out, gnn_input = self.gnnloader
                    result = {
                        'n_id_in': n_id_in,
                        'n_id_out': n_id_out,
                        'gnn_input': gnn_input,
                        'lm_batch_id': None,
                    }
                    yield result
                else:
                    n_id_in, n_id_out, gnn_input = next(sampler_iter)
                    assert torch.all(n_id_in[:len(n_id_out)] == n_id_out)
                    result = {
                        'n_id_in': n_id_in,
                        'n_id_out': n_id_out,
                        'gnn_input': gnn_input,
                        'lm_batch_id': None,
                    }
                    yield result

            except StopIteration:
                break

    def __len__(self) -> int:
        return self.num_samples
