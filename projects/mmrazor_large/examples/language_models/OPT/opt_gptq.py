# Copyright (c) OpenMMLab. All rights reserved.
# Example for opt is converted from https://github.com/ist-daslab/sparsegpt
import torch
from datautils import get_loaders
from transformers import OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from utils import opt_eval, opt_infer

from mmrazor.implementations.pruning.sparse_gpt.utils import \
    memory_efficient_forward
from mmrazor.implementations.quantization.gptq import (GPTQLinear,
                                                       TritonGPTQLinear)
from mmrazor.utils import print_log


def enable_observer_linear(model):
    print_log('Enable updating qparams for GPTQLinear!')
    for _, module in model.named_modules():
        if isinstance(module, GPTQLinear):
            module.fix_qparams = False


def disable_observer_linear(model):
    print_log('Disable updating qparams for GPTQLinear!')
    for _, module in model.named_modules():
        if isinstance(module, GPTQLinear):
            module.fix_qparams = True


def del_redundant_attr(model):
    print_log('Del redundant weight for GPTQLinear!')
    for _, module in model.named_modules():
        if isinstance(module, TritonGPTQLinear):
            del module.weight


def get_model(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='Llama model to load')
    parser.add_argument(
        '--dataset',
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
        default=16,
        help='Batchsize for calibration and evaluation.')
    parser.add_argument(
        '--save', type=str, default='', help='Path to saved model.')
    parser.add_argument(
        '--quant_ckpt', type=str, default='', help='Quantized ckpt to load.')
    parser.add_argument(
        '--dev', type=str, default='cuda:0', help='Use which device.')
    parser.add_argument(
        '-m',
        type=bool,
        default=False,
        help='Whether to enable memory efficient forward')

    args = parser.parse_args()

    DEV = args.dev

    model = get_model(args.model)
    model.to(DEV)
    model.eval()
    print_log('load model over')

    from mmrazor.implementations.quantization import gptq
    compressor = gptq.GPTQCompressor()
    # use_triton_ops is True
    compressor.prepare(
        model.model.layers,
        quant_conv=True,
        use_triton_ops=True,
        quant_linear=True,
        bits=4,
        groupsize=128)

    # # # quantize activation for linear
    # # a_qconfig = dict(bits=4, perchannel=False, sym=False)
    # compressor.prepare(
    #     model.model.decoder,
    #     quant_conv=True,
    #     quant_linear=True,
    #     use_triton_ops=False,
    #     # a_qconfig=a_qconfig
    # )

    if args.quant_ckpt:
        del_redundant_attr(model)
        model.load_state_dict(torch.load(args.quant_ckpt))
    else:
        dataloader, testloader = get_loaders(
            args.dataset,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen)
        print_log('load data for infer over')

        compressor.init_hessian()
        enable_observer_linear(model)
        with memory_efficient_forward(
                model, wrap_modules=[OPTDecoderLayer], enabled=args.m,
                device=DEV):
            compressor.register_hessian_hooks()
            opt_infer(
                model,
                testloader,
                DEV,
                batch_size=args.batch_size,
                num_samples=args.nsamples)
            compressor.remove_hessian_hooks()
            compressor.quant_with_default_qconfig(device=DEV)

    disable_observer_linear(model)
    with memory_efficient_forward(
            model, wrap_modules=[OPTDecoderLayer], enabled=args.m, device=DEV):

        # for dataset in ['wikitext2', 'ptb', 'c4']:
        for dataset in ['wikitext2']:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
            print_log(dataset)
            opt_eval(model, testloader, DEV, batch_size=args.batch_size)

    if args.save and not args.quant_ckpt:
        print_log(f'save model in {args.save}')
        torch.save(model.state_dict(), args.save)
