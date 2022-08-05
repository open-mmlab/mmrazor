import torch

def is_symmetric_quant(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]

def sync_tensor(tensor):
    global USE_LINK
    global USE_DDP
    if USE_LINK:
        if tensor.is_cuda is True:
            tensor.data = tensor.data / link.get_world_size()
            link.allreduce(tensor.data)
    elif USE_DDP:
        tensor.data = tensor.data / dist.get_world_size()
        dist.all_reduce(tensor.data)
    return tensor


def pot_quantization(tensor: torch.Tensor, mode='round'):
    log2t = torch.log2(tensor)
    if mode == 'round':
        log2t = (torch.round(log2t) - log2t).detach() + log2t
    else:
        assert mode == 'floor' 
        log2t = (torch.floor(log2t) - log2t).detach() + log2t
    return 2 ** log2t