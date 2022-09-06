# Customize NAS algorithms

Here we show how to develop new NAS algorithms with an example of SPOS.

1. Register a new algorithm

Create a new file `mmrazor/models/algorithms/nas/spos.py`, class `SPOS` inherits from class `BaseAlgorithm`

```Python
from mmrazor.registry import MODELS
from ..base import BaseAlgorithm

@MODELS.register_module()
class SPOS(BaseAlgorithm):
    def __init__(self, **kwargs):
        super(SPOS, self).__init__(**kwargs)
        pass

    def loss(self, batch_inputs, data_samples):
        pass
```

2. Develop new algorithm components (optional)

SPOS can directly use class `OneShotModuleMutator` as core functions provider. If mutators provided in MMRazor don’t meet your needs, you can develop new algorithm components for your algorithm like `OneShotModuleMutator`, we will take `OneShotModuleMutator` as an example to introduce how to develop a new algorithm component:

a. Create a new file `mmrazor/models/mutators/module_mutator/one_shot_module_mutator.py`, class `OneShotModuleMutator` inherits from class `ModuleMutator`

b. Finish the functions you need in `OneShotModuleMutator`, eg: `sample_choices`, `set_choices` and so on.

```Python
from mmrazor.registry import MODELS
from .module_mutator import ModuleMutator


@MODELS.register_module()
class OneShotModuleMutator(ModuleMutator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample_choices(self) -> Dict[int, Any]:
        pass

    def set_choices(self, choices: Dict[int, Any]) -> None:
        pass

    @property
    def mutable_class_type(self):
        return OneShotMutableModule
```

c. Import the new mutator

You can either add the following line to `mmrazor/models/mutators/__init__.py`

```Python
from .module_mutator import OneShotModuleMutator
```

or alternatively add

```Python
custom_imports = dict(
    imports=['mmrazor.models.mutators.module_mutator.one_shot_module_mutator'],
    allow_failed_imports=False)
```

to the config file to avoid modifying the original code.

d. Use the algorithm component in your config file

```Python
mutator=dict(type='mmrazor.OneShotModuleMutator')
```

For further information, please refer to [Mutator ](https://aicarrier.feishu.cn/docx/doxcnmcie75HcbqkfBGaEoemBKg)for more details.

3. Rewrite its `loss` function.

Develop key logic of your algorithm in function`loss`. When having special steps to optimize, you should rewrite the function `train_step`.

```Python
@MODELS.register_module()
class SPOS(BaseAlgorithm):
    def __init__(self, **kwargs):
        super(SPOS, self).__init__(**kwargs)
        pass

    def sample_subnet(self):
        pass

    def set_subnet(self, subnet):
        pass

    def loss(self, batch_inputs, data_samples):
        if self.is_supernet:
            random_subnet = self.sample_subnet()
            self.set_subnet(random_subnet)
            return self.architecture(batch_inputs, data_samples, mode='loss')
        else:
            return self.architecture(batch_inputs, data_samples, mode='loss')
```

4. Add your custom functions (optional)

After finishing your key logic in function `loss`, if you also need other custom functions, you can add them in class `SPOS` as follows.

5. Import the class

You can either add the following line to `mmrazor/models/algorithms/nas/__init__.py`

```Python
from .spos import SPOS

__all__ = ['SPOS']
```

or alternatively add

```Python
custom_imports = dict(
    imports=['mmrazor.models.algorithms.nas.spos'],
    allow_failed_imports=False)
```

to the config file to avoid modifying the original code.

6. Use the algorithm in your config file

```Python
model = dict(
    type='mmrazor.SPOS',
    architecture=supernet,
    mutator=dict(type='mmrazor.OneShotModuleMutator'))
```
