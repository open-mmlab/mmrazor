# Algorithm

## Introduction

### What is algorithm in MMRazor

MMRazor is a model compression toolkit, which includes 4 mianstream technologies:

- Neural Architecture Search (NAS)
- Pruning
- Knowledge Distillation (KD)
- Quantization (come soon)

And in MMRazor, `algorithm` is a general item for these technologies. For example, in NAS,

[SPOS](https://github.com/open-mmlab/mmrazor/blob/master/configs/nas/spos)[ ](https://arxiv.org/abs/1904.00420)is an `algorithm`,  [CWD](https://github.com/open-mmlab/mmrazor/blob/master/configs/distill/cwd) is also an `algorithm` of knowledge distillation.

`algorithm`  is the entrance of `mmrazor/models` . Its role in MMRazor is the same as both `classifier` in [MMClassification](https://github.com/open-mmlab/mmclassification) and `detector` in [MMDetection](https://github.com/open-mmlab/mmdetection).

### About base algorithm

In the directory of `models/algorithms`, all model compression algorithms are divided into 4 subdirectories: nas / pruning / distill / quantization. These algorithms must inherit from `BaseAlgorithm`, whose definition is as below.

```Python
from typing import Dict, List, Optional, Tuple, Union
from mmengine.model import BaseModel
from mmrazor.registry import MODELS

LossResults = Dict[str, torch.Tensor]
TensorResults = Union[Tuple[torch.Tensor], torch.Tensor]
PredictResults = List[BaseDataElement]
ForwardResults = Union[LossResults, TensorResults, PredictResults]

@MODELS.register_module()
class BaseAlgorithm(BaseModel):

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 init_cfg: Optional[Dict] = None):

        ......

        super().__init__(data_preprocessor, init_cfg)
        self.architecture = architecture

    def forward(self,
                batch_inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor') -> ForwardResults:

        if mode == 'loss':
            return self.loss(batch_inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(batch_inputs, data_samples)
        elif mode == 'predict':
            return self._predict(batch_inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def loss(
        self,
        batch_inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> LossResults:
        """Calculate losses from a batch of inputs and data samples."""
        return self.architecture(batch_inputs, data_samples, mode='loss')

    def _forward(
        self,
        batch_inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> TensorResults:
        """Network forward process."""
        return self.architecture(batch_inputs, data_samples, mode='tensor')

    def _predict(
        self,
        batch_inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> PredictResults:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        return self.architecture(batch_inputs, data_samples, mode='predict')
```

As you can see from above, `BaseAlgorithm` is inherited from `BaseModel` of MMEngine. `BaseModel` implements the basic functions of the algorithmic model, such as weights initialize,

batch inputs preprocess (see more information in `BaseDataPreprocessor` class of MMEngine), parse losses, and update model parameters. For more details of `BaseModel` , you can see docs for `BaseModel`.

`BaseAlgorithm`'s forward is just a wrapper of `BaseModel`'s forward. Sub-classes inherited from BaseAlgorithm only need to override the `loss` method, which implements the logic to calculate loss, thus various algorithms can be trained in the runner.

## How to use existing algorithms in MMRazor

1. Configure your architecture that will be slimmed

- Use the model config of other repos of OpenMMLab directly as below, which is an example of setting Faster-RCNN as our architecture.

```Python
_base_ = [
    'mmdet::_base_/models/faster_rcnn_r50_fpn.py',
]

architecture = _base_.model
```

- Use your customized model as below, which is an example of defining a VGG model as our architecture.

```{note}
How to customize architectures can refer to our tutorial: [Customize Architectures](https://mmrazor.readthedocs.io/en/main/advanced_guides/customize_architectures.html).
```

```Python
default_scope='mmcls'
architecture = dict(
    type='ImageClassifier',
    backbone=dict(type='VGG', depth=11, num_classes=1000),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
```

2. Apply the registered algorithm to your architecture.

```{note}
The arg name of `algorithm` in config is **model** rather than **algorithm** in order to get better supports of MMCV and MMEngine.
```

Maybe more args in model need to set according to the used algorithm.

```Python
model = dict(
    type='BaseAlgorithm',
    architecture=architecture)
```

```{note}
About the usage of `Config`, refer to [config.md](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/config.md) please.
```

3. Apply some custom hooks or loops to your algorithm. (optional)

- Custom hooks

```Python
custom_hooks = [
    dict(type='NaiveVisualizationHook', priority='LOWEST'),
]
```

- Custom loops

```Python
_base_ = ['./spos_shufflenet_supernet_8xb128_in1k.py']

# To chose from ['train_cfg', 'val_cfg', 'test_cfg'] based on your loop type
train_cfg = dict(
    _delete_=True,
    type='mmrazor.EvolutionSearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator)

val_cfg = dict()
test_cfg = dict()
```

## How to customize your algorithm

### Common pipeline

1. Register a new algorithm

Create a new file `mmrazor/models/algorithms/{subdirectory}/xxx.py`

```Python
from mmrazor.models.algorithms import BaseAlgorithm
from mmrazor.registry import MODELS

@MODELS.register_module()
class XXX(BaseAlgorithm):
    def __init__(self, architecture):
        super().__init__(architecture)
        pass

    def loss(self, batch_inputs):
        pass
```

2. Rewrite its `loss` method.

```Python
from mmrazor.models.algorithms import BaseAlgorithm
from mmrazor.registry import MODELS

@MODELS.register_module()
class XXX(BaseAlgorithm):
    def __init__(self, architecture):
        super().__init__(architecture)
        ......

    def loss(self, batch_inputs):
        ......
        return LossResults
```

3. Add the remaining functions of the algorithm

```{note}
This step is special because of the diversity of algorithms. Some functions of the algorithm may also be implemented in other files.
```

```Python
from mmrazor.models.algorithms import BaseAlgorithm
from mmrazor.registry import MODELS

@MODELS.register_module()
class XXX(BaseAlgorithm):
    def __init__(self, architecture):
        super().__init__(architecture)
        ......

    def loss(self, batch_inputs):
        ......
        return LossResults

    def aaa(self):
        ......

    def bbb(self):
        ......
```

4. Import the class

You can add the following line to `mmrazor/models/algorithms/{subdirectory}/__init__.py`

```CoffeeScript
from .xxx import XXX

__all__ = ['XXX']
```

In addition, import XXX in `mmrazor/models/algorithms/__init__.py`

5. Use the algorithm in your config file.

Please refer to the previous section about how to use existing algorithms in MMRazor

```Python
model = dict(
    type='XXX',
    architecture=architecture)
```

### Pipelines for different algorithms

Please refer to our tutorials about how to customize different algorithms for more details as below.

1. NAS

[Customize NAS algorithms](https://mmrazor.readthedocs.io/en/main/advanced_guides/customize_nas_algorithms.html)

2. Pruning

[Customize Pruning algorithms](https://mmrazor.readthedocs.io/en/main/advanced_guides/customize_pruning_algorithms.html)

3. Distill

[Customize KD algorithms](https://mmrazor.readthedocs.io/en/main/advanced_guides/customize_kd_algorithms.html)
