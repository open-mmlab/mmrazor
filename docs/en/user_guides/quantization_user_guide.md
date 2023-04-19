# Quantization

## Introduction

MMRazor's quantization is OpenMMLab's quantization toolkit, which has got through task models and model deployment. With its help, we can quantize and deploy pre-trained models in OpenMMLab to specified backend quickly. Of course, it can also contribute to implementing some custom quantization algorithms easier.

### Major features

- **Ease of use**. Benefited from PyTorch fx, we can quantize our model without modifying the original model, but with user-friendly config.
- **Multiple backends deployment support**. Because of the specificity of each backend, a gap in performance usually exists between before and after deployment. We provided some common backend deployment support to reduce the gap as much.
- **Multiple task repos support.** Benefited from OpenMMLab 2.0, our quantization can support all task repos of OpenMMLab without extra code.
- **Be compatible with PyTorch's core module in quantization**. Some core modules in PyTorch can be used directly in mmrazor, such as `Observer`, `FakeQuantize`, `BackendConfig` and so on.

## Quick run

```{note}
MMRazor's quantization is based on `torch==1.13`. Other requirements are the same as MMRazor's
```

Model quantization is in mmrazor, but quantized model deployment is in mmdeploy. So we need to the another branches as follows if we need to delopy our quantized model:

mmdeploy:  https://github.com/open-mmlab/mmdeploy/tree/for_mmrazor

```{note}
If you try to compress mmdet's models and have used `dense_heads`, you can use this branch:
https://github.com/HIT-cwh/mmdetection/tree/for_mmrazor to avoid the problem that some code can not be traced by `torch.fx.tracer`.
```

1. Quantize the float model in mmrazor.

```Shell
# For QAT (Quantization Aware Training)
python tools/train.py ${CONFIG_PATH} [optional arguments]

# For PTQ (Post-training quantization)
python tools/ptq.py ${CONFIG_PATH} [optional arguments]
```

2. Evaluate the quantized model. (optional)

```Shell
python tools/test.py ${CONFIG_PATH} ${CHECKPOINT_PATH}
```

3. Export quantized model to a specific backend in mmdeploy. (required by model deployment)

```Shell
# MODEL_CFG_PATH is the used config in mmrazor.
python ./tools/deploy.py \
    ${DEPLOY_CFG_PATH} \
    ${MODEL_CFG_PATH} \
    ${MODEL_CHECKPOINT_PATH} \
    ${INPUT_IMG} \
    [optional arguments]
```

This step is the same as how to export an OpenMMLab model to a specific backend. For more details, please refer to [How to convert model](https://github.com/open-mmlab/mmdeploy/blob/master/docs/en/02-how-to-run/convert_model.md)

4. Evaluate the quantized backend model. (optional)

```Shell
python tools/test.py \
    ${DEPLOY_CFG} \
    ${MODEL_CFG} \
    --model ${BACKEND_MODEL_FILES} \
    [optional arguments]
```

This step is the same as evaluating backend models. For more details, please refer to [How to evaluate model](https://github.com/open-mmlab/mmdeploy/blob/master/docs/en/02-how-to-run/profile_model.md)

## How to quantize your own model quickly

If you want to try quantize your own model quickly, you just need to learn about how to change our provided config.

**Case 1: If the model you want to quantize is in our provided configs.**

You can refer to the previous chapter Quick Run.

**Case 2: If the model you want to quantize is not in our provided configs.**

Let us take `resnet50` as an example to show how to handle case 2.

```Python
_base_ = [
    'mmcls::resnet/resnet18_8xb32_in1k.py',
    '../../deploy_cfgs/mmcls/classification_openvino_dynamic-224x224.py'
]

val_dataloader = dict(batch_size=32)

test_cfg = dict(
    type='mmrazor.PTQLoop',
    calibrate_dataloader=val_dataloader,
    calibrate_steps=32,
)

global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=True),
    a_qscheme=dict(
        qdtype='quint8', bit=8, is_symmetry=True, averaging_constant=0.1),
)

float_checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth'  # noqa: E501

model = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor=dict(
        type='mmcls.ClsDataPreprocessor',
        num_classes=1000,
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        to_rgb=True),
    architecture=_base_.model,
    deploy_cfg=_base_.deploy_cfg,
    float_checkpoint=float_checkpoint,
    quantizer=dict(
        type='mmrazor.OpenVINOQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ])))

model_wrapper_cfg = dict(type='mmrazor.MMArchitectureQuantDDP', )
```

This is a config that quantize `resnet18` with OpenVINO backend. You just need to modify two args: `_base_` and `float_checkpoint`.

```Python
# before
_base_ = ['mmcls::resnet/resnet18_8xb32_in1k.py']
float_checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth'

# after
_base_ = ['mmcls::resnet/resnet50_8xb32_in1k.py']
float_checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
```

- `_base_` will be called from mmcls by mmengine, so you can just use mmcls provided configs directly. Other repos are similar.
- `float_checkpoint ` is a pre-trained float checkpoint by OpenMMLab. You can find it in the corresponding repo.

After modifying required config, we can use it the same as case 1.

## How to improve your quantization performance

If you can not be satisfied with quantization performance by applying our provided configs to your own model, you can try to improve it with our provided various quantization schemes by modifying `global_qconfig`.

```Python
global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=True),
    a_qscheme=dict(
        qdtype='quint8', bit=8, is_symmetry=True, averaging_constant=0.1),
)
```

As shown above, `global_qconfig` contains server common core args as follows:

- Observes

In `forward`, they will update the statistics of the observed Tensor. And they should provide a `calculate_qparams` function that computes the quantization parameters given the collected statistics.

```{note}
Whether it is per channel quantization depends on whether `PerChannel` is in the observer name.
```

Because mmrazor's quantization has been compatible with PyTorch's observers, we can use observers in PyTorch and our custom observers.

Supported observers list in Pytorch.

```Python
FixedQParamsObserver
HistogramObserver
MinMaxObserver
MovingAverageMinMaxObserver
MovingAveragePerChannelMinMaxObserver
NoopObserver
ObserverBase
PerChannelMinMaxObserver
PlaceholderObserver
RecordingObserver
ReuseInputObserver
UniformQuantizationObserverBase
```

- Fake quants

In `forward`, they will update the statistics of the observed Tensor and fake quantize the input. They should also provide a `calculate_qparams` function that computes the quantization parameters given the collected statistics.

Because mmrazor's quantization has been compatible with PyTorch's fakequants, we can use fakequants in PyTorch and our custom fakequants.

Supported fakequants list in Pytorch.

```Python
FakeQuantize
FakeQuantizeBase
FixedQParamsFakeQuantize
FusedMovingAvgObsFakeQuantize
```

- Qschemes

Include some basic quantization configurations.

`qdtype`: to specify whether quantized data type is sign or unsign. It can be chosen from \[ 'qint8',  'quint8' \]

```{note}
If your model need to be deployed, `qdtype` must be consistent with the dtype in the corresponding backendconfig. Otherwise fakequant will not be inserted in front of the specified OPs.

backendconfigs dir:
mmrazor/mmrazor/structures/quantization/backend_config
```

`bit`: to specify the quantized data bit. It can be chosen from \[1 ~ 16\].

`is_symmetry`: to specify whether to use symmetry quantization. It can be chosen from \[ True, False \]

The specified qscheme is actually implemented by observers, so how to configurate other args needs to be based on the given observers, such as `is_symmetric_range` and `averaging_constant`.

## How to customize your quantization algorithm

If you try to customize your quantization algorithm, you can refer to the following link for more details.

[Customize Quantization algorithms](https://github.com/open-mmlab/mmrazor/blob/quantize/docs/en/advanced_guides/customize_quantization_algorithms.md)
