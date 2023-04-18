# Post-Training Quantization (PTQ)

> [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295)

<!-- [ALGORITHM] -->

## Abstract

While neural networks have advanced the frontiers in many applications, they often come at a high computational cost. Reducing the power and latency of neural network inference is key if we want to integrate modern networks into edge devices with strict power and compute requirements. Neural network quantization is one of the most effective ways of achieving these savings but the additional noise it induces can lead to accuracy degradation. In this white paper, we introduce state-of-the-art algorithms for mitigating the impact of quantization noise on the network's performance while maintaining low-bit weights and activations. We start with a hardware motivated introduction to quantization and then consider two main classes of algorithms: Post-Training Quantization (PTQ) and Quantization-Aware-Training (QAT). PTQ requires no re-training or labelled data and is thus a lightweight push-button approach to quantization. In most cases, PTQ is sufficient for achieving 8-bit quantization with close to floating-point accuracy. QAT requires fine-tuning and access to labeled training data but enables lower bit quantization with competitive results. For both solutions, we provide tested pipelines based on existing literature and extensive experimentation that lead to state-of-the-art performance for common deep learning models and tasks.

## Results and models

### Classification

| Model        | Dataset  | Backend  | Top 1 Acc（fp32） | Top 1 Acc（int8） | Top 1 Acc（deployed） | Config                                                      | Download                                                                                                                                                                                                                                                                                                       |
| ------------ | -------- | -------- | --------------- | --------------- | ------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| resnet18     | ImageNet | openvino | 69.90           | 69.742          | 69.74               | [config](./ptq_openvino_resnet18_8xb32_in1k_calib32xb32.py) | [model](https://download.openmmlab.com/mmrazor/v1/quantization/ptq/openvino/ptq_openvino_resnet18_8xb32_in1k_calib32xb32_20230330_163655-2386d965.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/quantization/ptq/openvino/ptq_openvino_resnet18_8xb32_in1k_calib32xb32_20230330_163655-2386d965.log) |
| resnet50     | ImageNet | openvino | 76.55           | 76.374          | 76.378              | [config](./ptq_openvino_resnet50_8xb32_in1k_calib32xb32.py) | [model](https://download.openmmlab.com/mmrazor/v1/quantization/ptq/openvino/ptq_openvino_resnet50_8xb32_in1k_calib32xb32_20230330_170115-2acd6014.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/quantization/ptq/openvino/ptq_openvino_resnet50_8xb32_in1k_calib32xb32_20230330_170115-2acd6014.log) |
| mobilenet_v2 | ImageNet | openvino | 71.86           | 70.224          | 70.292              | [config](./ptq_openvino_mbv2_8xb32_in1k_calib32xb32.py)     | [model](https://download.openmmlab.com/mmrazor/v1/quantization/ptq/openvino/ptq_openvino_mbv2_8xb32_in1k_calib32xb32_20230330_170909-364822ad.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/quantization/ptq/openvino/ptq_openvino_mbv2_8xb32_in1k_calib32xb32_20230330_170909-364822ad.log)         |
| resnet18     | ImageNet | tensorrt | 69.90           | 69.762          | 69.85               | [config](./ptq_tensorrt_resnet18_8xb32_in1k_calib32xb32.py) | [model](https://download.openmmlab.com/mmrazor/v1/quantization/ptq/tensorrt/ptq_tensorrt_resnet18_8xb32_in1k_calib32xb32_20230331_144323-640b272e.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/quantization/ptq/tensorrt/ptq_tensorrt_resnet18_8xb32_in1k_calib32xb32_20230331_144323-640b272e.log) |
| resnet50     | ImageNet | tensorrt | 76.55           | 76.372          | 76.374              | [config](./ptq_tensorrt_resnet50_8xb32_in1k_calib32xb32.py) | [model](https://download.openmmlab.com/mmrazor/v1/quantization/ptq/tensorrt/ptq_tensorrt_resnet50_8xb32_in1k_calib32xb32_20230331_145011-d2da300f.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/quantization/ptq/tensorrt/ptq_tensorrt_resnet50_8xb32_in1k_calib32xb32_20230331_145011-d2da300f.log) |
| mobilenet_v2 | ImageNet | tensorrt | 71.86           | 70.324          | 70.548              | [config](./ptq_tensorrt_mbv2_8xb32_in1k_calib32xb32.py)     | [model](https://download.openmmlab.com/mmrazor/v1/quantization/ptq/tensorrt/ptq_tensorrt_mbv2_8xb32_in1k_calib32xb32_20230331_153131-335988e4.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/quantization/ptq/tensorrt/ptq_tensorrt_mbv2_8xb32_in1k_calib32xb32_20230331_153131-335988e4.log)         |

### Detection

| Model          | Dataset | Backend  | box AP（fp32） | box AP（int8） | box AP（deployed） | Config                                                         | Download                 |
| -------------- | ------- | -------- | ------------ | ------------ | ---------------- | -------------------------------------------------------------- | ------------------------ |
| retina_r50_fpn | COCO    | openvino | 36.5         | 36.3         | 36.3             | [config](./ptq_openvino_retina_r50_1x_coco_calib32xb32.py)     | [model](<>) \| [log](<>) |
| yolox_s        | COCO    | openvino | 40.5         | 38.5         | 38.5             | [config](./ptq_openvino_yolox_s_8xb8-300e_coco_calib32xb32.py) | [model](<>) \| [log](<>) |
| retina_r50_fpn | COCO    | tensorrt | 36.5         | 36.2         | 36.3             | [config](./ptq_tensorrt_retina_r50_1x_coco_calib32xb32.py)     | [model](<>) \| [log](<>) |
| yolox_s        | COCO    | tensorrt | 40.5         | 38.8         | 39.3             | [config](./ptq_tensorrt_yolox_s_8xb8-300e_coco_calib32xb32.py) | [model](<>) \| [log](<>) |

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

**PTQ for pretrain model**

```
python tools/ptq.py ${CONFIG}
```

**Test for quantized model**

```
python tools/test.py ${CONFIG} ${CKPT}
```

For more details, please refer to [Quantization User Guide](https://mmrazor.readthedocs.io/en/main/user_guides/quantization_user_guide.html)
