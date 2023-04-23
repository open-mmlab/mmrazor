# Copyright (c) OpenMMLab. All rights reserved.
# Example for opt is converted from https://github.com/ist-daslab/sparsegpt
import torch
import torch.nn as nn
from transformers import OPTForCausalLM

from mmrazor.utils import print_log


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
             batch_size=16):
    print_log('Evaluating ...')

    seqlen = model.seqlen

    testenc: torch.Tensor = testenc.input_ids  # type: ignore # 1, N
    testenc = fold_tokens(testenc, seqlen)  # B N

    use_cache = model.config.use_cache
    model.config.use_cache = False
    nlls = []

    for i, batch in enumerate(torch.split(testenc, batch_size)):
        B = batch.shape[0]

        batch = batch.to(dev)
        out: torch.Tensor = model(batch)[0]  # 1

        shift_logits = out[:, :-1, :].contiguous().flatten(0, 1)  # (B N) C
        shift_labels = batch[:, 1:].flatten()  # (B N)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels)
        neg_log_likelihood = loss.float() * seqlen * B
        nlls.append(neg_log_likelihood)

        print_log(f'{(i+1)*batch_size} / {len(testenc)}')

    ppl = torch.exp(torch.stack(nlls).sum() / (testenc.numel()))
    print_log(f'Perplexity: {ppl.item():3f}')
    model.config.use_cache = use_cache


@torch.no_grad()
def opt_infer(
    model: OPTForCausalLM,
    testenc,
    dev,
    batch_size=16,
    num_samples=128,
):
    print_log('Infer ...')

    seqlen = model.seqlen

    testenc: torch.Tensor = testenc.input_ids  # type: ignore # 1, N
    testenc = fold_tokens(testenc, seqlen)  # B N

    model.config.use_cache = False

    for i, batch in enumerate(torch.split(testenc, batch_size)):
        batch = batch.to(dev)
        _ = model(batch)[0]  # 1
        print_log(f'{(i+1)*batch_size} / {num_samples}')

        if (i + 1) * batch_size >= num_samples:
            break


class init_on_meta:

    def __init__(self, enable=True) -> None:
        self.enable = enable
        self.default_device = torch.ones([]).device

    def __enter__(self):
        if self.enable:
            torch.set_default_device('meta')

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enable:
            torch.set_default_device(self.default_device)
