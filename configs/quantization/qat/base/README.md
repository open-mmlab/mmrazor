# Quantization-Aware-Training (QAT)

> [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295)

<!-- [ALGORITHM] -->

## Abstract

While neural networks have advanced the frontiers in many applications, they often come at a high computational cost. Reducing the power and latency of neural network inference is key if we want to integrate modern networks into edge devices with strict power and compute requirements. Neural network quantization is one of the most effective ways of achieving these savings but the additional noise it induces can lead to accuracy degradation. In this white paper, we introduce state-of-the-art algorithms for mitigating the impact of quantization noise on the network's performance while maintaining low-bit weights and activations. We start with a hardware motivated introduction to quantization and then consider two main classes of algorithms: Post-Training Quantization (PTQ) and Quantization-Aware-Training (QAT). PTQ requires no re-training or labelled data and is thus a lightweight push-button approach to quantization. In most cases, PTQ is sufficient for achieving 8-bit quantization with close to floating-point accuracy. QAT requires fine-tuning and access to labeled training data but enables lower bit quantization with competitive results. For both solutions, we provide tested pipelines based on existing literature and extensive experimentation that lead to state-of-the-art performance for common deep learning models and tasks.

## Results and models

### Classification

| Model    | Dataset  | Backend  | Top 1 Acc（fp32） | Top 1 Acc（int8） | Config                                              | Download                                                                                                                                                                                                                                                                                       |
| -------- | -------- | -------- | --------------- | --------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| resnet18 | ImageNet | openvino | 69.90           | 69.98           | [config](./qat_openvino_resnet18_10e_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmrazor/v1/quantization/qat/openvino/qat_openvino_resnet18_8xb32_10e_in1k_20230413_172732-5b9ff01d.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/quantization/qat/openvino/qat_openvino_resnet18_8xb32_10e_in1k_20230413_172732-5b9ff01d.log) |

## Citation

```latex
 @misc{Nagel_Fournarakis_Amjad_Bondarenko_Baalen_Blankevoort_2021,
    title={A White Paper on Neural Network Quantization},
    journal={Cornell University - arXiv},
    author={Nagel, Markus and Fournarakis, Marios and Amjad, RanaAli and Bondarenko, Yelysei and Baalen, Martvan and Blankevoort, Tijmen},
    year={2021},
    month={Jun}
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
