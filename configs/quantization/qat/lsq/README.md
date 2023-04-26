# Learned Step Size Quantization (LSQ)

> [Learned Step Size Quantization](https://arxiv.org/abs/1902.08153)

<!-- [ALGORITHM] -->

## Abstract

Deep networks run with low precision operations at inference time offer power and space advantages over high precision alternatives, but need to overcome the challenge of maintaining high accuracy as precision decreases. Here, we present a method for training such networks, Learned Step Size Quantization, that achieves the highest accuracy to date on the ImageNet dataset when using models, from a variety of architectures, with weights and activations quantized to 2-, 3- or 4-bits of precision, and that can train 3-bit models that reach full precision baseline accuracy. Our approach builds upon existing methods for learning weights in quantized networks by improving how the quantizer itself is configured. Specifically, we introduce a novel means to estimate and scale the task loss gradient at each weight and activation layer's quantizer step size, such that it can be learned in conjunction with other network parameters. This approach works using different levels of precision as needed for a given system and requires only a simple modification of existing training code.

## Results and models

### Classification

| Model    | Dataset  | Backend  | Top 1 Acc（fp32） | Top 1 Acc（int8） | Max Epochs | Config                                               | Download                                                                                                                                                                                                                                                                                         |
| -------- | -------- | -------- | --------------- | --------------- | ---------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| resnet18 | ImageNet | openvino | 69.90           | 69.418          | 10         | [config](./lsq_openvino_resnet18_8xb32_10e_in1k.py)  | [model](https://download.openmmlab.com/mmrazor/v1/quantization/qat/openvino/lsq_openvino_resnet18_8xb32_10e_in1k_20230413_224237-36eac1f1.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/quantization/qat/openvino/lsq_openvino_resnet18_8xb32_10e_in1k_20230413_224237-36eac1f1.log)   |
| resnet18 | ImageNet | openvino | 69.90           | 69.992          | 100        | [config](./lsq_openvino_resnet18_8xb32_100e_in1k.py) | [model](https://download.openmmlab.com/mmrazor/v1/quantization/qat/openvino/lsq_openvino_resnet18_8xb32_100e_in1k_20230402_173316-ca5993bf.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/quantization/qat/openvino/lsq_openvino_resnet18_8xb32_100e_in1k_20230402_173316-ca5993bf.log) |

## Citation

```latex
 @misc{Esser_McKinstry_Bablani_Appuswamy_Modha_2019,
    title={Learned Step Size Quantization},
    journal={arXiv: Learning},
    author={Esser, StevenK. and McKinstry, JeffreyL. and Bablani, Deepika and Appuswamy, Rathinakumar and Modha, DharmendraS.},
    year={2019},
    month={Feb}
 }
```

## Getting Started

**QAT for pretrain model**

```
python tools/train.py ${CONFIG}
```

**Test for quantized model**

```
python tools/test.py ${CONFIG} ${CKPT}
```

For more details, please refer to [Quantization User Guide](https://mmrazor.readthedocs.io/en/main/user_guides/quantization_user_guide.html)
