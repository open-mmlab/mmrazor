<div align="center">
  <img src="../../resources/mmrazor-logo.png" width="600"/>
</div>

# MMRazor Examples for Large Models

## Introduction

MMRazor is dedicated to the development of general-purpose model compression tools. Now, MMRazor not only supports conventional CV model compression but also extends to support large models. This project will provide examples of MMRazor's compression for various large models, including LLama, stable diffusion, and more.

## Installation

```shell
pip install openmim
mim install mmcv
mim install mmengine
pip install git+https://github.com/open-mmlab/mmrazor.git
git clone github.com/open-mmlab/mmrazor-examples.git
```

## Model-Algorithm Example Matrix

|                                               | ResNet                                                                    | OPT                                                                         | LLama                                                                         | Stable diffusion |
| --------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ---------------- |
| [SparseGPT](examples/algorithms/SparseGPT.md) | [:white_check_mark:](examples/model_examples/ResNet/sparse_gpt/README.md) | [:white_check_mark:](examples/model_examples/language_models/OPT/README.md) | [:white_check_mark:](examples/model_examples/language_models/Llama/README.md) |                  |

## PaperList

We provide a paperlist for researchers in the field of model compression for large models. If you want to add your paper to this list, please submit a PR.

| Paper     | Title                                                                                                                 | Type    | MMRazor                                                |
| --------- | --------------------------------------------------------------------------------------------------------------------- | ------- | ------------------------------------------------------ |
| SparseGPT | [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)           | Pruning | [:white_check_mark:](examples/algorithms/SparseGPT.md) |
| GPTQ      | [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) | Quant   |                                                        |
