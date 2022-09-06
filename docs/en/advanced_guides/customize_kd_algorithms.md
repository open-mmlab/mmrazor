# Customize KD algorithms

Here we show how to develop new KD algorithms with an example of `SingleTeacherDistill`.

1. Register a new algorithm

Create a new file `mmrazor/models/algorithms/distill/configurable/single_teacher_distill.py`, class `SingleTeacherDistill` inherits from class `BaseAlgorithm`

```Python
from mmrazor.registry import MODELS
from ..base import BaseAlgorithm

@ALGORITHMS.register_module()
class SingleTeacherDistill(BaseAlgorithm):
    def __init__(self, use_gt, **kwargs):
        super(Distillation, self).__init__(**kwargs)
        pass

    def train_step(self, data, optimizer):
        pass
```

2. Develop connectors (Optional) .

Take ConvModuleConnector as an example.

```python
from mmrazor.registry import MODELS
from .base_connector import BaseConnector

@MODELS.register_module()
class ConvModuleConncetor(BaseConnector):
    def __init__(self, in_channel, out_channel, kernel_size = 1, stride = 1):
        ...

    def forward_train(self, feature):
        ...
```

3. Develop distiller.

Take `ConfigurableDistiller` as an example.

```python
from .base_distiller import BaseDistiller
from mmrazor.registry import MODELS


@MODELS.register_module()
class ConfigurableDistiller(BaseDistiller):
    def __init__(self,
                 student_recorders = None,
                 teacher_recorders = None,
                 distill_deliveries = None,
                 connectors = None,
                 distill_losses = None,
                 loss_forward_mappings = None):
        ...

     def build_connectors(self, connectors):
        ...

     def build_distill_losses(self, losses):
        ...

     def compute_distill_losses(self):
        ...
```

4. Develop custom loss (Optional).

Here we take `L1Loss` as an example. Create a new file in `mmrazor/models/losses/l1_loss.py`.

```python
from mmrazor.registry import MODELS

@MODELS.register_module()
class L1Loss(nn.Module):
    def __init__(
        self,
        loss_weight: float = 1.0,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = 'mean',
    ) -> None:
        super().__init__()
        ...

    def forward(self, s_feature, t_feature):
        loss = F.l1_loss(s_feature, t_feature, self.size_average, self.reduce,
                         self.reduction)
        return self.loss_weight * loss
```

5. Import the class

You can either add the following line to `mmrazor/models/algorithms/__init__.py`

```Python
from .single_teacher_distill import SingleTeacherDistill

__all__ = [..., 'SingleTeacherDistill']
```

or alternatively add

```Python
custom_imports = dict(
    imports=['mmrazor.models.algorithms.distill.configurable.single_teacher_distill'],
    allow_failed_imports=False)
```

to the config file to avoid modifying the original code.

6. Use the algorithm in your config file

```Python
algorithm = dict(
    type='Distill',
    distiller=dict(type='SingleTeacherDistill', ...),
    # you can also use your new algorithm components here
    ...
)
```
