# Example for opt is converted from https://github.com/ist-daslab/sparsegpt
import torch
import torch.nn as nn
from transformers import OPTForCausalLM

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


@torch.no_grad()
def opt_eval(model: OPTForCausalLM,
             testenc,
             dev,
             dataset: str,
             log_wandb: bool = False):
    print('Evaluating ...')

    testenc: torch.Tensor = testenc.input_ids  # type: ignore
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    nlls = []

    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):(i + 1) * model.seqlen].to(dev)
        out = model(batch)[0]  # 1

        shift_logits = out[:, :-1, :].contiguous()  # 1 N C
        shift_labels = batch[:, 1:]  # 1 N

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f'Perplexity: {ppl.item():3f}')
    model.config.use_cache = use_cache


@torch.no_grad()
def opt_infer(
    model: OPTForCausalLM,
    testenc,
    dev,
    num_samples=128,
):
    print('Infer ...')

    testenc: torch.Tensor = testenc.input_ids  # type: ignore
    nsamples = testenc.numel() // model.seqlen

    model.config.use_cache = False

    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):(i + 1) * model.seqlen].to(dev)
        _ = model(batch)[0]  # 1

        if i > num_samples:
            break


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
    model = model.cuda()
    print('load model over')
    DEV = torch.device('cuda:0')

    dataloader, testloader = get_loaders(
        'c4', seed=args.seed, model=args.model, seqlen=model.seqlen)

    from mmrazor.implementations.pruning import sparse_gpt
    mutator = sparse_gpt.SparseGptMutator.init_from_a_model(model)

    mutator.start_init_hessian()
    opt_infer(model, testloader, DEV, num_samples=128)
    mutator.end_init_hessian()
    mutator.prune_24()

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
        print(dataset)
        opt_eval(model, testloader, DEV, dataset)
