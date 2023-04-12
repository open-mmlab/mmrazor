# Example for opt is converted from https://github.com/ist-daslab/sparsegpt
import torch
import torch.nn as nn
from transformers import OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTDecoderLayer

from mmrazor.implementations.pruning.sparse_gpt.utils import \
    memory_efficient_forward

has_wandb = False


def get_opt(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = OPTForCausalLM.from_pretrained(
        model,
        torch_dtype='auto',
        mirror='https://mirror.nju.edu.cn/hugging-face-models',
        local_files_only=True)
    model.seqlen = model.config.max_position_embeddings
    return model


def fold_tokens(tokens: torch.Tensor, batch_seq_len=2048):
    # tokens: 1 N
    N = tokens.shape[1]
    num_drop = N % batch_seq_len
    if num_drop != 0:
        tokens = tokens[:, :-num_drop]
    tokens = tokens.reshape([-1, batch_seq_len])  # B N
    return tokens


@torch.no_grad()
def opt_eval(model: OPTForCausalLM,
             testenc,
             dev=torch.device('cuda:0'),
             batch_size=64,
             log_wandb: bool = False):
    print('Evaluating ...')

    seqlen = model.seqlen

    testenc: torch.Tensor = testenc.input_ids  # type: ignore # 1, N
    testenc = fold_tokens(testenc, seqlen)  # B N

    use_cache = model.config.use_cache
    model.config.use_cache = False
    nlls = []

    for batch in torch.split(testenc, batch_size):
        B = batch.shape[0]

        batch = batch.to(dev)
        out: torch.Tensor = model(batch)[0]  # 1

        shift_logits = out[:, :-1, :].contiguous().flatten(0, 1)  # (B N) C
        shift_labels = batch[:, 1:].flatten()  # (B N)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels)
        neg_log_likelihood = loss.float() * seqlen * B
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (testenc.numel()))
    print(f'Perplexity: {ppl.item():3f}')
    model.config.use_cache = use_cache


@torch.no_grad()
def opt_infer(
    model: OPTForCausalLM,
    testenc,
    dev,
    batch_size=64,
):
    print('Infer ...')

    seqlen = model.seqlen

    testenc: torch.Tensor = testenc.input_ids  # type: ignore # 1, N
    testenc = fold_tokens(testenc, seqlen)  # B N

    model.config.use_cache = False

    for batch in torch.split(testenc, batch_size):
        batch = batch.to(dev)
        _ = model(batch)[0]  # 1


if __name__ == '__main__':
    import argparse

    from datautils import get_loaders

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
    args = parser.parse_args()

    model = get_opt(args.model)
    model.eval()
    print('load model over')
    DEV = torch.device('cuda:0')

    dataloader, testloader = get_loaders(
        'c4', seed=args.seed, model=args.model, seqlen=model.seqlen)

    from mmrazor.implementations.pruning import sparse_gpt
    mutator = sparse_gpt.SparseGptMutator.init_from_a_model(
        model.model.decoder)

    with memory_efficient_forward(model, wrap_modules=[OPTDecoderLayer]):

        mutator.start_init_hessian()
        opt_infer(model, testloader, DEV)
        mutator.end_init_hessian()
        mutator.prune_24()

        for dataset in ['wikitext2', 'ptb', 'c4']:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
            print(dataset)
            opt_eval(model, testloader, DEV)
