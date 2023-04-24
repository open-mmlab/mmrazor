# Customize Quantization algorithms

Here we show how to develop new QAT algorithms with an example of LSQ on OpenVINO backend.

This document is mainly aimed at QAT because the ptq process is relatively fixed and the components we provide can meet most of the needs. We will first give an overview of the overall required development components, and then introduce the specific implementation step by step.

## Overall

In the mmrazor quantization pipeline, in order to better support the openmmlab environment, we have configured most of the code modules for users. You can configure all the components directly in the config file. How to configure them can be found in our [file](https://github.com/open-mmlab/mmrazor/blob/quantize/configs/quantization/qat/minmax_openvino_resnet18_8xb32_in1k.py).

```Python
global_qconfig = dict(
    w_observer=dict(),
    a_observer=dict(),
    w_fake_quant=dict(),
    a_fake_quant=dict(),
    w_qscheme=dict(),
    a_qscheme=dict(),
)
model = dict(
    type='mmrazor.MMArchitectureQuant',
    architecture=resnet,
    quantizer=dict(
        type='mmrazor.OpenvinoQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict()))
train_cfg = dict(type='mmrazor.LSQEpochBasedLoop')
```

For `algorithm` and `tracer`, we recommend that you use the default configurations `MMArchitectureQuant` and `CustomTracer` provided by us. These two module operators are specially built for the openmmlab environment, while other modules can refer to the following steps and choose or develop new operators according to your needs.

To adapt to different backends, you need to select a different `quantizer`.

To develop new quantization algorithms, you need to define new `observer` and `fakequant`.

If the existing `loop` does not meet your needs, you may need to make some changes to the existing `loop` based on your algorithm.

## Detailed steps

1. Select a quantization algorithm

We recommend that you directly use the`MMArchitectureQuant` in `mmrazor/models/algorithms/quantization/mm_architecture.py`.The class `MMArchitectureQuant` inherits from class `BaseAlgorithm`.

This structure is built for the model in openmmlab. If you have other requirements, you can also refer to this [document](https://mmrazor.readthedocs.io/en/main/advanced_guides/customize_architectures.html#develop-common-model-components) to design the overall framework.

2. Select quantizer

At present, the quantizers we support are `NativeQuantizer`, `OpenVINOQuantizer`, `TensorRTQuantizer` and `AcademicQuantizer` in `mmrazor/models/quantizers/`. `AcademicQuantizer` and `NativeQuantizer` inherit from class `BaseQuantizer` in `mmrazor/models/quantizers/base.py`:

```Python
class BaseQuantizer(BaseModule):
    def __init__(self, tracer):
        super().__init__()
        self.tracer = TASK_UTILS.build(tracer)
    @abstractmethod
    def prepare(self, model, graph_module):
        """tmp."""
        pass
    def swap_ff_with_fxff(self, model):
        pass
```

`NativeQuantizer` is the operator we developed to adapt to the environment of mmrazor according to pytorch's official quantization logic. `AcademicQuantizer` is an operator designed for academic research to give users more space to operate.

The class `OpenVINOQuantizer` and `TensorRTQuantizer` inherits from class `NativeQuantize`. They adapted `OpenVINO` and `TensorRT`backend respectively. You can also try to develop a quantizer based on other backends according to your own needs.

3. Select tracer

Tracer we use `CustomTracer` in `mmrazor/models/task_modules/tracer/fx/custom_tracer.py`. You can inherit this class and customize your own tracer.

4. Develop new fakequant method(optional)

You can use fakequants provided by pytorch in `mmrazor/models/fake_quants/torch_fake_quants.py` as core functions provider. If you want to use the fakequant methods from other papers, you can also define them yourself. Let's take lsq as an example as follows:

a.Create a new file `mmrazor/models/fake_quants/lsq.py`, class `LearnableFakeQuantize` inherits from class `FakeQuantizeBase`.

b. Finish the functions you need, eg: `observe_quant_params`, `calculate_qparams` and so on.

```Python
from mmrazor.registry import MODELS
from torch.ao.quantization import FakeQuantizeBase

@MODELS.register_module()
class LearnableFakeQuantize(FakeQuantizeBase):
    def __init__(self,
                observer,
                quant_min=0,
                quant_max=255,
                scale=1.,
                zero_point=0.,
                use_grad_scaling=True,
                zero_point_trainable=False,
                **observer_kwargs):
        super(LearnableFakeQuantize, self).__init__()
        pass

    def observe_quant_params(self):
        pass

    def calculate_qparams(self):
        pass

    def forward(self, X):
        pass
```

c.Import the module in `mmrazor/models/fake_quants/__init__.py`.

```Python
from .lsq import LearnableFakeQuantize

__all__ = ['LearnableFakeQuantize']
```

5. Develop new observer(optional)

You can directly use observers provided by pytorch in `mmrazor/models/observers/torch_observers.py` or use observers customized by yourself. Let's take `LSQObserver` as follows:

a.Create a new observer file `mmrazor/models/observers/lsq.py`, class `LSQObserver` inherits from class `MinMaxObserver` and `LSQObserverMixIn`. These two observers can calculate `zero_point` and `scale`, respectively.

b.Finish the functions you need, eg: `calculate_qparams` and so on.

```Python
from mmrazor.registry import MODELS
from torch.ao.quantization.observer import MinMaxObserver

class LSQObserverMixIn:
    def __init__(self):
        self.tensor_norm = None

    @torch.jit.export
    def _calculate_scale(self):
        scale = 2 * self.tensor_norm / math.sqrt(self.quant_max)
        sync_tensor(scale)
        return scale

@MODELS.register_module()
class LSQObserver(MinMaxObserver, LSQObserverMixIn):
    """LSQ observer.
    Paper: Learned Step Size Quantization. <https://arxiv.org/abs/1902.08153>
    """
    def __init__(self, *args, **kwargs):
        MinMaxObserver.__init__(self, *args, **kwargs)
        LSQObserverMixIn.__init__(self)

    def forward(self, x_orig):
        """Records the running minimum, maximum and tensor_norm of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        self.tensor_norm = x.abs().mean()
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        """Calculates the quantization parameters."""
        _, zero_point = MinMaxObserver.calculate_qparams(self)
        scale = LSQObserverMixIn._calculate_scale(self)
        return scale, zero_point
```

c.Import the module in `mmrazor/models/observers/__init__.py`

```Python
from .lsq import LSQObserver

__all__ = ['LSQObserver']
```

6. Select loop or develop new loop

At present, the QAT loops we support are `PTQLoop` and `QATEpochBasedLoop`, in `mmrazor/engine/runner/quantization_loops.py`. We can develop a new `LSQEpochBasedLoop` inherits from class `QATEpochBasedLoop` and finish the functions we need in LSQ method.

```Python
from mmengine.runner import EpochBasedTrainLoop

@LOOPS.register_module()
class LSQEpochBasedLoop(QATEpochBasedLoop):
    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            freeze_bn_begin: int = -1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(
            runner,
            dataloader,
            max_epochs,
            val_begin,
            val_interval,
            freeze_bn_begin=freeze_bn_begin,
            dynamic_intervals=dynamic_intervals)

        self.is_first_batch = True

    def prepare_for_run_epoch(self):
        pass

    def prepare_for_val(self):
        pass

    def run_epoch(self) -> None:
        pass
```

And then Import the module in `mmrazor/engine/runner/__init__.py`

```Python
from .quantization_loops import LSQEpochBasedLoop

__all__ = ['LSQEpochBasedLoop']
```

7. Use the algorithm in your config file

After completing the above steps, we have all the components of the qat algorithm, and now we can combine them in the config file.

a.First,  `_base_` stores the location of the model that needs to be quantized.

b.Second, configure observer,fakequant and qscheme in `global_qconfig` in detail.
You can configure the required quantization bit width and quantization methods in `qscheme`, such as symmetric quantization or asymmetric quantization.

c.Third, build the whole mmrazor model in `model`.

d.Finally, complete all the remaining required configuration files.

```Python
_base_ = ['mmcls::resnet/resnet18_8xb16_cifar10.py']

global_qconfig = dict(
    w_observer=dict(type='mmrazor.LSQPerChannelObserver'),
    a_observer=dict(type='mmrazor.LSQObserver'),
    w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=True),
    a_qscheme=dict(qdtype='quint8', bit=8, is_symmetry=True),
)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='MMArchitectureQuant',
    data_preprocessor=dict(
        type='mmcls.ClsDataPreprocessor',
        num_classes=1000,
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        to_rgb=True),
    architecture=resnet,
    float_checkpoint=float_ckpt,
    quantizer=dict(
        type='mmrazor.OpenVINOQuantizer',
        is_qat=True,
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ])))

# learning policy
optim_wrapper = dict()
param_scheduler = dict()
model_wrapper_cfg = dict()

# train, val, test setting
train_cfg = dict(type='mmrazor.LSQEpochBasedLoop')
val_cfg = dict()
test_cfg = val_cfg
```
