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
from utils import init_on_meta, opt_eval_fsdp, opt_infer_fsdp

from mmrazor.implementations.pruning import sparse_gpt
from mmrazor.utils import print_log


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print_log(f'init {rank}/{world_size}', only_rank0=False)


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

            name2p = dict(module_copy.named_parameters(remove_duplicate=False))
            for n, p in module.named_parameters():
                if '_flat_param' not in n:
                    n = n.replace('_fsdp_wrapped_module.', '')
                    try:
                        p.data.copy_(name2p[n])
                    except Exception:
                        pass
            name2p = dict(module_copy.named_buffers(remove_duplicate=False))
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
        mutator = sparse_gpt.SparseGptCompressor()
        mutator.prepare(model.model.decoder)
        return model, mutator

    with init_on_meta(enable=True):
        model, mutator = build()

    if rank == 0:
        model_copy, _ = build()  # init on cpu
    else:
        model_copy = None

    # init fsdp
    size_based_auto_wrap_policy_x = functools.partial(
        size_based_auto_wrap_policy, min_num_params=int(1e8))

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

    mutator.init_hessian(device='cuda')
    mutator.register_hessian_hooks(model)

    _, testloader = get_loaders(
        args.dataset, seed=args.seed, model=model_name, seqlen=model.seqlen)
    testloader = build_language_loader(
        testloader, world_size, rank, model, batch_size=batch_size)
    opt_infer_fsdp(model, testloader)

    mutator.remove_hessian_hooks()

    # prune
    name2module = dict(model.named_modules())
    module2name = {}
    module2name = dict([(v, k) for k, v in name2module.items()])

    with torch.no_grad():
        for fsdp in FSDP.fsdp_modules(model):
            fsdp._reset_lazy_init()
            with FSDP.summon_full_params(fsdp, recurse=False):
                fsdp_name = module2name[fsdp]
                for name, op in fsdp.named_modules():
                    if name.count('_fsdp_wrapped_module') <= 1:
                        if isinstance(op, sparse_gpt.SparseGptMixIn):
                            try:
                                op.prune(0.5, prunen=2, prunem=4)
                                print_log(
                                    f'prune {fsdp_name}.{name} successfully.',  # noqa
                                    only_rank0=True)
                            except Exception as e:
                                print_log(
                                    f'prune {fsdp_name}.{name} failed, as {e}',  # noqa
                                    only_rank0=True)
            fsdp._reset_lazy_init()

    # save
    if args.save:
        print_log(f'save model in {args.save}')
        model._reset_lazy_init()
        with FSDP.summon_full_params(model, rank0_only=True, writeback=False):
            if dist.get_rank() == 0:
                model.save_pretrained(args.save)

    # val
    torch.cuda.empty_cache()
    model._reset_lazy_init()
    for dataset in ['wikitext2', 'ptb', 'c4']:
        _, testloader = get_loaders(
            dataset, seed=args.seed, model=model_name, seqlen=model.seqlen)
        testloader = build_language_loader(
            testloader, world_size, rank, model, batch_size=batch_size)
        print_log(dataset)
        opt_eval_fsdp(model, testloader, torch.device('cuda'))


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
        '--save', type=str, default='', help='Path to saved model.')
    parser.add_argument(
        '--world_size', type=int, default=1, help='Number of GPUs to use.')
    args = parser.parse_args()

    WORLD_SIZE = args.world_size
    mp.spawn(main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
