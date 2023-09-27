from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Sized, Iterator, NamedTuple
from collections.abc import Mapping

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch_sparse import SparseTensor
import dgl
from dgl.heterograph import DGLBlock
from torch_geometric.data import Data
import torch.nn.functional as F

from tqdm import tqdm

from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, has_length, denumpify_detensorize, ShardedDDPOption
from transformers.deepspeed import deepspeed_init
from transformers.trainer import logger, is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import find_batch_size, nested_concat, nested_numpify, nested_truncate, IterableDatasetShard, get_parameter_names

from bert.bert_gnn_sampler import RecursiveSampler, OneshotSampler, DictWrapper
from bert.bert_utils import TinyData



class GLBaseTrainer(Trainer):
    def __init__(self, gnn_train_loader_func, gnn_eval_loader, semi_supervised_info, lr_scale, decay_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gnn_train_loader_func = gnn_train_loader_func
        self.gnn_eval_loader = gnn_eval_loader

        self.lr_scale = lr_scale
        self.decay_scale = decay_scale

        self.semi_supervised_info = semi_supervised_info
        # assert 'labels' in self.semi_supervised_info
        # assert 'train_mask' in self.semi_supervised_info
        # assert 'val_mask' in self.semi_supervised_info
        # assert 'test_mask' in self.semi_supervised_info
        assert 'pre_gnn_bz' in self.semi_supervised_info
        assert 'admm_gnn_bz' in self.semi_supervised_info
        assert 'valid_test_dp' in self.semi_supervised_info
        assert 'whole_data' in self.semi_supervised_info
        self.self_learning = False

        self.phase = None

    def train(self, phase, *args, **kwargs,):
        self.phase = phase
        super().train(*args, **kwargs,)
    
    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, DGLBlock):
            kwargs = dict(device=self.args.device)
            if self.deepspeed and data.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
            return data.to(**kwargs)

        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.args.device)
            if self.deepspeed and data.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
            return data.to(**kwargs)
        return data
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and 'gnn_model' in n],
                    "weight_decay": self.decay_scale*self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and 'gnn_model' not in n],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.lr_scale*self.args.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and 'gnn_model' in n],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and 'gnn_model' not in n],
                    "weight_decay": 0.0,
                    "lr": self.lr_scale*self.args.learning_rate,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def evaluation_loop_gnn(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.per_device_eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        if args.past_index >= 0:
            self._past = None

        # Will be useful when we have an iterable dataset so don't know its length.

        all_preds, all_labels = self.pred_gnn(model, args)
        self.model.all_logits = all_preds

        train_mask, val_mask, test_mask = self.get_split()
        metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels), train_mask, val_mask, test_mask,)

        self.model.preprocess_feat_label(self.semi_supervised_info, all_preds=None if not self.self_learning else all_preds) # self_learning = True: start training

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        if train_mask is not None:
            if self.semi_supervised_info['labels'].dim() == 1:
                model.label_emb[~train_mask] = torch.softmax(all_preds / model.pseudo_temp, dim=-1).clone()[~train_mask]
            else:
                model.label_emb[~train_mask] = torch.sigmoid(all_preds / model.pseudo_temp).clone()[~train_mask]

        model.is_augmented = True
        # if model.label_as_feat:
        #     raise NotImplementedError

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=len(all_labels))
    

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        
        if self.phase in ['pre_lm']:
            return super().evaluation_loop(dataloader,
                    description,
                    prediction_loss_only,
                    ignore_keys,
                    metric_key_prefix) # lm_batch_size
        elif self.phase in ['pre_gnn']:
            return self.evaluation_loop_gnn(dataloader,
                    description,
                    prediction_loss_only,
                    ignore_keys,
                    metric_key_prefix)
        else:
            raise NotImplementedError

    def get_eval_dataloader(self, eval_dataset = None) -> DataLoader:
        return super().get_eval_dataloader(eval_dataset)

    def get_train_dataloader(self) -> DataLoader:
        if self.phase in ['pre_lm']:
            if self.semi_supervised_info['valid_test_dp'] > 0.0:
                self.train_dataset = TinyData(self.semi_supervised_info['whole_data'], self.semi_supervised_info['train_mask'], self.semi_supervised_info['valid_test_dp'])
            return super().get_train_dataloader() # lm_batch_size
        elif self.phase in ['pre_gnn']:
            return DictWrapper(self.gnn_train_loader_func(self.semi_supervised_info['pre_gnn_bz'])) # gnn_batch_size
        else:
            raise NotImplementedError

    def pred_gnn(self, model, args):
        self.callback_handler.eval_dataloader = self.gnn_eval_loader
        all_preds = []
        all_labels = []
        for n_id_in, n_id_out, batch in self.gnn_eval_loader:
            with torch.no_grad():
                batch = self._prepare_input(batch)
                _, pred, labels = model.gnn_inference(n_id_in, n_id_out, batch)
            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return all_preds, all_labels
 
    def get_split(self,):
        train_mask = self.semi_supervised_info['train_mask']
        val_mask = self.semi_supervised_info['val_mask']
        test_mask = self.semi_supervised_info['test_mask']
        return train_mask, val_mask, test_mask

    
    def get_test_nodedataloader(self, test_dataset) -> DataLoader:
        data_collator = None

        if isinstance(test_dataset, torch.utils.data.IterableDataset):
            return DataLoader(
                test_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def update_emb(self, dataset):
        model = self._wrap_model(self.model, training=False)
        
        # # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # # while ``train`` is running, cast it to the right dtype first and then put on device

        model.eval()

        dataloader = self.get_test_nodedataloader(dataset)
        # Main evaluation loop
        self.callback_handler.eval_dataloader = dataloader
        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                model.update_emb(**inputs)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

    def load_hidden_state(self, load_dir, perm, save_gnn_path=None):

        if os.path.exists(os.path.join(load_dir, 'hidden_state.pt')):

            hidden_state = torch.load(os.path.join(load_dir, 'hidden_state.pt'))
            self.model.hist_emb.emb.copy_(hidden_state[perm])
            self.model.preprocess_feat_label(self.semi_supervised_info)

        else:
            self.phase = 'pre_lm'
            self.update_emb(self.semi_supervised_info['whole_data'])
            self.model.is_augmented = False
            os.makedirs(load_dir, exist_ok=True)
            hidden_state = self.model.hist_emb.emb.clone()
            hidden_state[perm] = self.model.hist_emb.emb
            torch.save(hidden_state, os.path.join(load_dir, 'hidden_state.pt'))
            if save_gnn_path is not None:
                torch.save(hidden_state, os.path.join(save_gnn_path, 'hidden_state.pt'))
            self.model.preprocess_feat_label(self.semi_supervised_info)
        if self.model.label_inverse:
            self.phase = 'pre_gnn'
            self.evaluate()
            self.model.inverse_label(self.semi_supervised_info['graph'])

        self.self_learning = True
        

