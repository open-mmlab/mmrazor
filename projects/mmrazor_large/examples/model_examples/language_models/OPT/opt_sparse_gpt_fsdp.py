import functools
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from datautils import build_language_loader, get_loaders
from opt_sparse_gpt import get_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader
from utils import init_on_meta

from mmrazor.implementations.pruning import sparse_gpt
from mmrazor.utils import print_log


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print_log(f'init {rank}/{world_size}', only_rank0=False)


@torch.no_grad()
def opt_eval(
        model: nn.Module,
        dataloader: DataLoader,
        dev=torch.device('cuda:0'),
):
    print_log('Evaluating ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    loss_sum = torch.zeros([1], device=dev)
    total_seq_len = torch.zeros([1], device=dev, dtype=torch.long)

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

        print_log(f'{(i+1)*B} / {len(dataloader.dataset)}', only_rank0=False)

    if dist.is_initialized():
        dist.all_reduce(loss_sum)
        dist.all_reduce(total_seq_len)

    ppl = torch.exp(loss_sum / total_seq_len)
    print_log(f'Perplexity: {ppl.item():3f}')
    model.config.use_cache = use_cache


@torch.no_grad()
def opt_infer(
        model: nn.Module,
        dataloader: DataLoader,
        dev=torch.device('cuda:0'),
):
    print_log('Infering ...')

    model.config.use_cache = False

    for i, batch in enumerate(dataloader):
        B, seq_len = batch.shape[:2]

        batch = batch.to(dev)
        model(batch)[0]  # 1

        print_log(f'{(i+1)*B} / {len(dataloader.dataset)}')


def init_fn_wrapper(model: nn.Module, model_copy: nn.Module):

    def find_module_in_model_copy(module: nn.Module):
        name2module = dict(model.named_modules())
        module2name = dict([(v, k) for k, v in name2module.items()])

        name = module2name[module]
        return dict(model_copy.named_modules())[name]

    def _materialize_meta_module(module: nn.Module, ):

        def meta_to_empty(p: torch.Tensor):
            if p.device == torch.device('meta'):
                return p.new_empty(p.shape, device='cpu')
            else:
                return p

        module._apply(meta_to_empty)

        if dist.get_rank() == 0:
            assert model_copy is not None
            module_copy = find_module_in_model_copy(module)

            name2p = dict(module_copy.named_parameters())
            for n, p in module.named_parameters():
                if '_flat_param' not in n:
                    n = n.replace('_fsdp_wrapped_module.', '')
                    try:
                        p.data.copy_(name2p[n])
                    except Exception:
                        pass
            for n, p in module.named_buffers():
                if '_flat_param' not in n:
                    n = n.replace('_fsdp_wrapped_module.', '')
                    try:
                        p.data.copy_(name2p[n])
                    except Exception:
                        pass

    return _materialize_meta_module


def main(rank, world_size=8, args=None):
    setup(rank, world_size)

    model_name = args.model
    batch_size = args.batch_size

    def build():
        model = get_model(model_name)

        # init mutator
        mutator = sparse_gpt.SparseGptMutator()
        mutator.prepare_from_supernet(model.model.decoder)
        mutator.keep_hessian_in_float()
        return model, mutator

    with init_on_meta(enable=True):
        model, mutator = build()

    if rank == 0:
        model_copy, _ = build()  # init on cpu
        model_copy.lm_head.weight = nn.Parameter(
            model_copy.lm_head.weight.float().clone().detach())
    else:
        model_copy = None

    # init fsdp
    size_based_auto_wrap_policy_x = functools.partial(
        size_based_auto_wrap_policy)

    model = FSDP(
        model,
        auto_wrap_policy=size_based_auto_wrap_policy_x,
        cpu_offload=CPUOffload(True),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=rank,
        param_init_fn=init_fn_wrapper(model, model_copy),
        sync_module_states=True)

    print_log(model)

    # init hessian

    mutator.start_init_hessian()

    _, testloader = get_loaders(
        'c4', seed=args.seed, model=model_name, seqlen=model.seqlen)
    testloader = build_language_loader(
        testloader, world_size, rank, model, batch_size=batch_size)
    opt_infer(model, testloader)

    mutator.end_init_hessian()

    # prune

    with torch.no_grad():

        total_num_op = 0
        for fsdp in FSDP.fsdp_modules(model):
            if len(FSDP.fsdp_modules(fsdp)) == 1:
                fsdp._reset_lazy_init()
                with FSDP.summon_full_params(fsdp):
                    num_op = 0
                    for name, op in fsdp.named_modules():
                        if isinstance(op, sparse_gpt.SparseGptMixIn):
                            if num_op % world_size == rank:
                                try:
                                    op.prune(0.5, prunen=2, prunem=4)
                                    print_log(
                                        f'prune {name} on rank:{rank} successfully.',  # noqa
                                        only_rank0=False)
                                except Exception as e:
                                    print_log(
                                        f'prune {name} on rank:{rank} failed, as {e}',  # noqa
                                        only_rank0=False)
                            num_op += 1
                    num_op = 0
                    for name, op in fsdp.named_modules():
                        if isinstance(op, sparse_gpt.SparseGptMixIn):
                            dist.broadcast(op.weight, num_op % world_size)
                            num_op += 1
                    total_num_op += num_op

                fsdp._reset_lazy_init()
                torch.cuda.empty_cache()
    # val
    torch.cuda.empty_cache()
    model._reset_lazy_init()
    for dataset in ['wikitext2', 'ptb', 'c4']:
        _, testloader = get_loaders(
            dataset, seed=1000, model=model_name, seqlen=model.seqlen)
        testloader = build_language_loader(
            testloader, world_size, rank, model, batch_size=batch_size)
        print_log(dataset)
        opt_eval(model, testloader, torch.device('cuda'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str, help='OPT model to load; pass `facebook/opt-X`.')
    parser.add_argument(
        'dataset',
        type=str,
        choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed for sampling the calibration data.')
    parser.add_argument(
        '--nsamples',
        type=int,
        default=128,
        help='Number of calibration data samples.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batchsize for calibration and evaluation.')
    parser.add_argument(
        '-m',
        type=bool,
        default=False,
        help='Init on meta device to save memory.')
    parser.add_argument(
        '--save', type=str, default='', help='Path to saved model.')
    parser.add_argument(
        '--world_size', type=int, default=1, help='Number of GPUs to use.')
    args = parser.parse_args()

    WORLD_SIZE = args.world_size
    mp.spawn(main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
