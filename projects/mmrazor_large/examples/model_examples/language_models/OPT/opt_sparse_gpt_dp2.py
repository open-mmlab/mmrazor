import functools
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from datautils import get_loaders
from opt_sparse_gpt import get_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DistributedSampler

from mmrazor.implementations.pruning import sparse_gpt
from mmrazor.utils import print_log


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def fold_tokens(tokens: torch.Tensor, batch_seq_len=2048):
    # tokens: 1 N
    N = tokens.shape[1]
    num_drop = N % batch_seq_len
    if num_drop != 0:
        tokens = tokens[:, :-num_drop]
    tokens = tokens.reshape([-1, batch_seq_len])  # B N
    return tokens


class LanguageDataset(TorchDataset):

    def __init__(self, seq: torch.Tensor, seq_len: int = 2048) -> None:
        super().__init__()
        # seq: 1, N
        self.seq_len = seq_len

        self.seq = fold_tokens(seq)  # B N

    def __len__(self) -> int:
        return self.seq.shape[0]

    def __getitem__(self, index):
        return self.seq[index]


@torch.no_grad()
def opt_eval(
        model: nn.Module,
        dataloader: DataLoader,
        dev=torch.device('cuda:0'),
):
    print_log('Evaluating ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    loss_sum = 0
    total_seq_len = 0

    for i, batch in enumerate(dataloader):
        B, seq_len = batch.shape[:2]

        batch = batch.to(dev)
        out: torch.Tensor = model(batch)[0]  # 1

        shift_logits = out[:, :-1, :].contiguous().flatten(0, 1)  # (B N) C
        shift_labels = batch[:, 1:].flatten()  # (B N)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels)

        neg_log_likelihood = loss.float() * seq_len * B
        total_seq_len += seq_len * B
        loss_sum += neg_log_likelihood

        print_log(f'{(i+1)*B} / {len(dataloader.dataset)}')

    assert isinstance(loss_sum, torch.Tensor), f'{type(loss_sum)}'
    if dist.is_initialized():
        dist.all_reduce(loss_sum)
        total_seq_len *= dist.get_world_size()
    ppl = torch.exp(loss_sum / total_seq_len)
    print_log(f'Perplexity: {ppl.item():3f}')
    model.config.use_cache = use_cache


def build_language_loader(testloader, world_size, rank, model):
    val_dataset = LanguageDataset(testloader.input_ids, seq_len=model.seqlen)
    distributed_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        sampler=distributed_sampler)
    return val_dataloader


def main(rank, world_size=8):
    print(f'init {rank}/{world_size}')
    setup(rank, world_size)

    model_name = 'facebook/opt-125m'
    model = get_model(model_name)

    # init mutator
    mutator = sparse_gpt.SparseGptMutator()
    mutator.prepare_from_supernet(model.model.decoder)

    # init fsdp
    size_based_auto_wrap_policy_x = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100)

    model = FSDP(
        model,
        auto_wrap_policy=size_based_auto_wrap_policy_x,
        cpu_offload=CPUOffload,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=rank)

    print_log(model)

    # val

    for dataset in ['wikitext2', 'ptb', 'c4']:
        _, testloader = get_loaders(
            dataset, seed=1000, model=model_name, seqlen=model.seqlen)
        testloader = build_language_loader(testloader, world_size, rank, model)
        print_log(dataset)
        opt_eval(model, testloader, torch.device('cuda'))


if __name__ == '__main__':
    pass
    WORLD_SIZE = 4
    mp.spawn(main, args=(WORLD_SIZE, ), nprocs=WORLD_SIZE, join=True)
