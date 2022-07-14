# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.logging import MMLogger
from mmengine.optim import OptimWrapper
from torch.nn import Module


def reinitialize_optim_wrapper_count_status(model: Module,
                                            optim_wrapper: OptimWrapper,
                                            accumulative_counts: int,
                                            verbose: bool = True) -> None:
    if verbose:
        logger = MMLogger.get_current_instance()
        logger.warning('Reinitialize count status of optim wrapper')

    original_max_iters = \
        optim_wrapper.message_hub.runtime_info['max_iters']
    new_max_iters = original_max_iters * accumulative_counts
    original_init_iters = \
        optim_wrapper.message_hub.runtime_info['iter']
    new_init_iters = original_init_iters * accumulative_counts

    if verbose:
        logger.info(f'original `init_iters`: {original_init_iters}, '
                    f'new `init_iters`: {new_init_iters}; '
                    f'orginal `max_iters`: {original_max_iters}, '
                    f'new `max_iters`: {new_max_iters}')

    optim_wrapper.initialize_count_status(
        model=model, init_counts=new_init_iters, max_counts=new_max_iters)
