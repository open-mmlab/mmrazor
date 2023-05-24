# GPTQ

> [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)

<!-- [ALGORITHM] -->

## Abstract

Generative Pre-trained Transformer models, known as GPT or OPT, set themselves apart through breakthrough performance across complex language modelling tasks, but also by their extremely high computational and storage costs. Specifically, due to their massive size, even inference for large, highly-accurate GPT models may require multiple performant GPUs, which limits the usability of such models. While there is emerging work on relieving this pressure via model compression, the applicability and performance of existing compression techniques is limited by the scale and complexity of GPT models. In this paper, we address this challenge, and propose GPTQ, a new one-shot weight quantization method based on approximate second-order information, that is both highlyaccurate and highly-efficient. Specifically, GPTQ can quantize GPT models with 175 billion parameters in approximately four GPU hours, reducing the bitwidth down to 3 or 4 bits per weight, with negligible accuracy degradation relative to the uncompressed baseline. Our method more than doubles the compression gains relative to previously-proposed one-shot quantization methods, preserving accuracy, allowing us for the first time to execute an 175 billion-parameter model inside a single GPU for generative inference. Moreover, we also show that our method can still provide reasonable accuracy in the extreme quantization regime, in which weights are quantized to 2-bit or even ternary quantization levels. We show experimentally that these improvements can be leveraged for end-to-end inference speedups over FP16, of around 3.25x when using high-end GPUs (NVIDIA A100) and 4.5x when using more cost-effective ones (NVIDIA A6000). The implementation is available at https://github.com/IST-DASLab/gptq.

## Usage

GPTQ is easy to use in mmrazor. You can use it like this:

```python
from mmrazor.implementations.quantization import gptq

# initial model, dataloaders
model
train_loader, test_loader

## init gptq compressor and prepare for quantization
compressor = gptq.GPTQCompressor()
compressor.prepare(model)

## get hessian matrix
compressor.init_hessian()
compressor.register_hessian_hooks()
infer(model, test_loader, num_samples=num_samples)
compressor.remove_hessian_hooks()

## quant
compressor.quant_with_default_qconfig()

## to a normal torch model
model = compressor.to_static_model(model)

```

## Full Examples

- [ResNet](../examples/ResNet/README.md)
- [LLaMA](../examples/language_models/LLaMA/README.md)

## Cite

```latex
 @misc{
  Frantar_Ashkboos_Hoefler_Alistarh_2022,
  title={GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers},
  author={Frantar, Elias and Ashkboos, Saleh and Hoefler, Torsten and Alistarh, Dan},
  year={2022},
  month={Oct},
  language={en-US}
}
```
