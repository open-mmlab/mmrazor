# Customize mixed algorithms

Here we show how to customize mixed algorithms with our algorithm components. We take [AutoSlim ](https://github.com/open-mmlab/mmrazor/tree/main/configs/pruning/mmcls/autoslim)as an example.

```{note}
**Why is AutoSlim a mixed algorithm?**

In [AutoSlim](https://github.com/open-mmlab/mmrazor/tree/main/configs/pruning/mmcls/autoslim), the sandwich rule and the inplace distillation will be introduced to enhance the training process, which is called as the slimmable training. The sandwich rule means that we train the model at smallest width, largest width and (n − 2) random widths, instead of n random widths. And the inplace distillation means that we use the predicted label of the model at the largest width as the training label for other widths, while for the largest width we use ground truth. So both the KD algorithm and the pruning algorithm are used in [AutoSlim](https://github.com/open-mmlab/mmrazor/tree/main/configs/pruning/mmcls/autoslim).
```

1. Register a new algorithm

Create a new file `mmrazor/models/algorithms/nas/autoslim.py`, class `AutoSlim` inherits from class `BaseAlgorithm`. You need to build the KD algorithm component (distiller) and the pruning algorithm component (mutator) because AutoSlim is a mixed algorithm.

```{note}
You can also inherit from the existing algorithm instead of `BaseAlgorithm` if your algorithm is similar to the existing algorithm.
```

```{note}
You can choose existing algorithm components in MMRazor, such as `OneShotChannelMutator` and `ConfigurableDistiller` in AutoSlim.

If these in MMRazor don't meet your needs, you can customize new algorithm components for your algorithm. Reference is as follows:

[Customize NAS algorithms](https://mmrazor.readthedocs.io/en/main/advanced_guides/customize_nas_algorithms.html)
[Customize Pruning algorithms](https://mmrazor.readthedocs.io/en/main/advanced_guides/customize_pruning_algorithms.html)
[Customize KD algorithms](https://mmrazor.readthedocs.io/en/main/advanced_guides/customize_kd_algorithms.html)
```

```Python
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union
import torch
from torch import nn

from mmrazor.models.distillers import ConfigurableDistiller
from mmrazor.models.mutators import OneShotChannelMutator
from mmrazor.registry import MODELS
from ..base import BaseAlgorithm

VALID_MUTATOR_TYPE = Union[OneShotChannelMutator, Dict]
VALID_DISTILLER_TYPE = Union[ConfigurableDistiller, Dict]

@MODELS.register_module()
class AutoSlim(BaseAlgorithm):
    def __init__(self,
                 mutator: VALID_MUTATOR_TYPE,
                 distiller: VALID_DISTILLER_TYPE,
                 architecture: Union[BaseModel, Dict],
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 num_random_samples: int = 2,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(architecture, data_preprocessor, init_cfg)
        self.mutator = self._build_mutator(mutator)
        # `prepare_from_supernet` must be called before distiller initialized
        self.mutator.prepare_from_supernet(self.architecture)

        self.distiller = self._build_distiller(distiller)
        self.distiller.prepare_from_teacher(self.architecture)
        self.distiller.prepare_from_student(self.architecture)

        ......

    def _build_mutator(self,
                       mutator: VALID_MUTATOR_TYPE) -> OneShotChannelMutator:
        """build mutator."""
        if isinstance(mutator, dict):
            mutator = MODELS.build(mutator)
        if not isinstance(mutator, OneShotChannelMutator):
            raise TypeError('mutator should be a `dict` or '
                            '`OneShotModuleMutator` instance, but got '
                            f'{type(mutator)}')

        return mutator

    def _build_distiller(
            self, distiller: VALID_DISTILLER_TYPE) -> ConfigurableDistiller:
        if isinstance(distiller, dict):
            distiller = MODELS.build(distiller)
        if not isinstance(distiller, ConfigurableDistiller):
            raise TypeError('distiller should be a `dict` or '
                            '`ConfigurableDistiller` instance, but got '
                            f'{type(distiller)}')

        return distiller
```

2. Implement the core logic in `train_step`

In `train_step`, both the `mutator` and the `distiller` play an important role. For example, `sample_subnet`, `set_max_subnet` and `set_min_subnet` are supported by the `mutator`, and the function of`distill_step` is mainly implemented by the `distiller`.

```Python
@MODELS.register_module()
class AutoSlim(BaseAlgorithm):

     ......

     def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:

        def distill_step(
                batch_inputs: torch.Tensor, data_samples: List[BaseDataElement]
        ) -> Dict[str, torch.Tensor]:
            ......

        ......

        batch_inputs, data_samples = self.data_preprocessor(data, True)

        total_losses = dict()
        for kind in self.sample_kinds:
            # update the max subnet loss.
            if kind == 'max':
                self.set_max_subnet()
                ......
                total_losses.update(add_prefix(max_subnet_losses, 'max_subnet'))
            # update the min subnet loss.
            elif kind == 'min':
                self.set_min_subnet()
                min_subnet_losses = distill_step(batch_inputs, data_samples)
                total_losses.update(add_prefix(min_subnet_losses, 'min_subnet'))
            # update the random subnets loss.
            elif 'random' in kind:
                self.set_subnet(self.sample_subnet())
                random_subnet_losses = distill_step(batch_inputs, data_samples)
                total_losses.update(
                    add_prefix(random_subnet_losses, f'{kind}_subnet'))

        return total_losses
```

3. Import the class

You can either add the following line to `mmrazor/models/algorithms/nas/__init__.py`

```Python
from .autoslim import AutoSlim

__all__ = ['AutoSlim']
```

or alternatively add

```Python
custom_imports = dict(
    imports=['mmrazor.models.algorithms.nas.autoslim'],
    allow_failed_imports=False)
```

to the config file to avoid modifying the original code.

4. Use the algorithm in your config file

```Python
model= dict(
    type='mmrazor.AutoSlim',
    architecture=...,
    mutator=dict(
        type='OneShotChannelMutator',
        ...),
    distiller=dict(
        type='ConfigurableDistiller',
        ...),
    ...)
```
