# Customize pruning algorithms

Here we show how to develop new Pruning algorithms with an example of AutoSlim.

1. Register a new algorithm

Create a new file `mmrazor/models/algorithms/prunning/autoslim.py`, class `AutoSlim` inherits from class `BaseAlgorithm`.

```Python
from mmrazor.registry import MODELS
from .base import BaseAlgorithm

@MODELS.register_module()
class AutoSlim(BaseAlgorithm):
    def __init__(self,
                 mutator,
                 distiller,
                 architecture,
                 data_preprocessor,
                 init_cfg = None,
                 num_samples = 2) -> None:
        super().__init__(**kwargs)
        pass

    def train_step(self, data, optimizer):
        pass
```

2. Develop new algorithm components (optional)

AutoSlim can directly use class `OneShotChannelMutator` as core functions provider. If it can not meet your needs, you can develop new algorithm components for your algorithm like `OneShotChannalMutator`. We will take `OneShotChannelMutator` as an example to introduce how to develop a new algorithm component:

a. Create a new file `mmrazor/models/mutators/channel_mutator/one_shot_channel_mutator.py`, class `OneShotChannelMutator` can inherits from `ChannelMutator`.

b. Finish the functions you need, eg: `build_search_groups`, `set_choices` , `sample_choices` and so on

```Python
from mmrazor.registry import MODELS
from .channel_mutator import ChannelMutator


@MODELS.register_module()
class OneShotChannelMutator(ChannelMutator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample_choices(self):
        pass

    def set_choices(self, choice_dict):
        pass

    # supernet is a kind of architecture in `mmrazor/models/architectures/`
    def build_search_groups(self, supernet):
        pass
```

c. Import the module in `mmrazor/models/mutators/channel_mutator/__init__.py`

```Python
from .one_shot_channel_mutator import OneShotChannelMutator

 __all__ = [..., 'OneShotChannelMutator']
```

3. Rewrite its train_step

Develop key logic of your algorithm in function`train_step`

```Python
from mmrazor.registry import MODELS
from ..base import BaseAlgorithm

@ALGORITHMS.register_module()
class AutoSlim(BaseAlgorithm):
    def __init__(self,
                 mutator,
                 distiller,
                 architecture,
                 data_preprocessor,
                 init_cfg = None,
                 num_samples = 2) -> None:
        super(AutoSlim, self).__init__(**kwargs)
        pass

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:

        def distill_step(
                batch_inputs: torch.Tensor, data_samples: List[BaseDataElement]
        ) -> Dict[str, torch.Tensor]:
            ...
            return subnet_losses

        batch_inputs, data_samples = self.data_preprocessor(data, True)

        total_losses = dict()
        self.set_max_subnet()
        with optim_wrapper.optim_context(
                self), self.distiller.teacher_recorders:  # type: ignore
            max_subnet_losses = self(batch_inputs, data_samples, mode='loss')
            parsed_max_subnet_losses, _ = self.parse_losses(max_subnet_losses)
            optim_wrapper.update_params(parsed_max_subnet_losses)
        total_losses.update(add_prefix(max_subnet_losses, 'max_subnet'))

        self.set_min_subnet()
        min_subnet_losses = distill_step(batch_inputs, data_samples)
        total_losses.update(add_prefix(min_subnet_losses, 'min_subnet'))

        for sample_idx in range(self.num_samples):
            self.set_subnet(self.sample_subnet())
            random_subnet_losses = distill_step(batch_inputs, data_samples)
            total_losses.update(
                add_prefix(random_subnet_losses,
                           f'random_subnet_{sample_idx}'))

        return total_losses
```

4. Add your custom functions (optional)

After finishing your key logic in function `train_step`, if you also need other custom functions, you can add them in class `AutoSlim`.

5. Import the class

You can either add the following line to `mmrazor/models/algorithms/__init__.py`

```Python
from .pruning import AutoSlim

__all__ = [..., 'AutoSlim']
```

Or alternatively add

```Python
custom_imports = dict(
    imports=['mmrazor.models.algorithms.pruning.autoslim'],
    allow_failed_imports=False)
```

to the config file to avoid modifying the original code.

6. Use the algorithm in your config file

```Python
model = dict(
    type='AutoSlim',
    architecture=...,
    mutator=dict(type='OneShotChannelMutator', ...),
    )
```
