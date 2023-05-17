<div align="center">
  <img src="../../resources/mmrazor-logo.png" width="600"/>
</div>

# MMRazor for Large Models

## Introduction

MMRazor is dedicated to the development of general-purpose model compression tools. Now, MMRazor not only supports conventional CV model compression but also extends to support large models. This project will provide examples of MMRazor's compression for various large models, including LLaMA, stable diffusion, and more.

Code structure overview about large models.

```
mmrazor
├── implementations           # core algorithm components
    ├── pruning
    └── quantization
projects
└── mmrazor_large
    ├── algorithms            # algorithms usage introduction
    └── examples              # examples for various models about algorithms
        ├── language_models
        │   ├── LLaMA
        │   └── OPT
        └── ResNet
```

## Model-Algorithm Example Matrix

|                                      | ResNet                                          | OPT                                                          | LLama                                                          | Stable diffusion |
| ------------------------------------ | ----------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------- | ---------------- |
| [SparseGPT](algorithms/SparseGPT.md) | [:white_check_mark:](examples/ResNet/README.md) | [:white_check_mark:](examples/language_models/OPT/README.md) | [:white_check_mark:](examples/language_models/LLaMA/README.md) |                  |
| [GPTQ](algorithms/GPTQ.md)           | [:white_check_mark:](examples/ResNet/README.md) | [:white_check_mark:](examples/language_models/OPT/README.md) | [:white_check_mark:](examples/language_models/LLaMA/README.md) |                  |

## PaperList

We provide a paperlist for researchers in the field of model compression for large models. If you want to add your paper to this list, please submit a PR.

| Paper     | Title                                                                                                                 | Type         | MMRazor                                       |
| --------- | --------------------------------------------------------------------------------------------------------------------- | ------------ | --------------------------------------------- |
| SparseGPT | [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)           | Pruning      | [:white_check_mark:](algorithms/SparseGPT.md) |
| GPTQ      | [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) | Quantization | [:white_check_mark:](algorithms/GPTQ.md)      |
