import torch
import torch.nn as nn

from mmrazor.implementations.pruning import sparse_gpt


def infer(model: nn.Module,
          dataloader: torch.utils.data.DataLoader,
          num_batchs=256):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        accumulate_batch = 0
        for x, _ in dataloader:
            x = x.to(device)
            model(x)
            B = x.shape[0]
            accumulate_batch += B
            if accumulate_batch > num_batchs:
                break


def sparse_model(model: nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 num_batchs=256):

    mutator = sparse_gpt.SparseGptMutator.init_from_a_model(model)
    mutator.start_init_hessian()
    infer(model, dataloader, num_batchs)
    mutator.end_init_hessian()
    mutator.prune_24()
    return model
