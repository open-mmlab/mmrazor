# Once-For-All

> [ONCE-FOR-ALL: TRAIN ONE NETWORK AND SPE- CIALIZE IT FOR EFFICIENT DEPLOYMENT](https://arxiv.org/abs/1908.09791)

<!-- [ALGORITHM] -->

## Abstract

We address the challenging problem of efficient inference across many devices and resource constraints, especially on edge devices. Conventional approaches either manually design or use neural architecture search (NAS) to find a specialized neural network and train it from scratch for each case, which is computationally prohibitive (causing CO2 emission as much as 5 cars’ lifetime Strubell et al. (2019)) thus unscalable. In this work, we propose to train a once-for-all (OFA) network that supports diverse architectural settings by decoupling training and search, to reduce the cost. We can quickly get a specialized sub-network by selecting from the OFA network without additional training. To efficiently train OFA networks, we also propose a novel progressive shrinking algorithm, a generalized pruning method that reduces the model size across many more dimensions than pruning (depth, width, kernel size, and resolution). It can obtain a surprisingly large number of sub- networks (> 1019) that can fit different hardware platforms and latency constraints while maintaining the same level of accuracy as training independently. On diverse edge devices, OFA consistently outperforms state-of-the-art (SOTA) NAS methods (up to 4.0% ImageNet top1 accuracy improvement over MobileNetV3, or same accuracy but 1.5× faster than MobileNetV3, 2.6× faster than EfficientNet w.r.t measured latency) while reducing many orders of magnitude GPU hours and CO2 emission. In particular, OFA achieves a new SOTA 80.0% ImageNet top-1 accuracy under the mobile setting (\<600M MACs). OFA is the winning solution for the 3rd Low Power Computer Vision Challenge (LPCVC), DSP classification track and the 4th LPCVC, both classification track and detection track.

## Get Started

We product inference models which are published by official Once-For-All repo and converted by MMRazor.

### Subnet test on ImageNet

```bash
CUDA_VISIBLE_DEVICES=0 PORT=29500 ./tools/dist_test.sh \
  configs/nas/mmcls/onceforall/ofa_mobilenet_subnet_8xb256_in1k.py \
  none 1 --work-dir $WORK_DIR \
  --cfg-options model.init_cfg.checkpoint=$OFA_CKPT model.init_weight_from_supernet=False
```

## Results and models

| Dataset  |       Supernet       |                                                           Subnet                                                            | Params(M) | Flops(G) | Top-1 |                                                             Config                                                              |                                                                             Download                                                                             |         Remarks         |
| :------: | :------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------: | :------: | :---: | :-----------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------: |
| ImageNet | AttentiveMobileNetV3 | [search space](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/configs/_base_/nas_backbones/ofa_mobilenetv3_supernet.py) |    7.6    |  747.8   | 77.5  | [config](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/configs/nas/mmcls/onceforall/ofa_mobilenet_supernet_32xb64_in1k.py) |                 [model](https://download.openmmlab.com/mmrazor/v1/ofa/ofa_mobilenet_supernet_d234_e346_k357_w1_0.py_20221214_0940-d0ebc66f.pth)                  | Converted from the repo |
| ImageNet | AttentiveMobileNetV3 |                                            note8_lat@22ms_top1@70.4_finetune@25                                             |    4.3    |   70.9   | 70.3  |                   [config](https://download.openmmlab.com/mmrazor/v1/ofa/rtmdet/OFA_SUBNET_NOTE8_LAT22.yaml)                    | [model](https://download.openmmlab.com/mmrazor/v1/ofa/ofa_mobilenet_subnet_8xb256_in1k_note8_lat%4022ms_top1%4070.4_finetune%4025.py_20221214_0938-fb7fb84f.pth) | Converted from the repo |
| ImageNet | AttentiveMobileNetV3 |                                            note8_lat@31ms_top1@72.8_finetune@25                                             |    4.6    |  105.4   | 72.6  |                   [config](https://download.openmmlab.com/mmrazor/v1/ofa/rtmdet/OFA_SUBNET_NOTE8_LAT31.yaml)                    | [model](https://download.openmmlab.com/mmrazor/v1/ofa/ofa_mobilenet_subnet_8xb256_in1k_note8_lat%4031ms_top1%4072.8_finetune%4025.py_20221214_0939-981a8b2a.pth) | Converted from the repo |

**Note**:

1. OFA provides a more fine-grained search mode, which searches expand ratios & kernel size for each block in every layer of the defined supernet, therefore the subnet configs (format as .yaml) is more complex than those of BigNAS/AttentiveNAS.
2. We product the [ofa script](../../../../tools/model_converters/convert_ofa_ckpt.py) to convert the official weight into MMRazor-style. The layer depth of a specific subnet is required when converting keys.
3. The models above are converted from the [once-for-all official repo](https://github.com/mit-han-lab/once-for-all). The config files of these models
   are only for inference. We don't ensure training accuracy of these config files and you are welcomed to contribute your reproduction results.

## Citation

```latex
@inproceedings{
  cai2020once,
  title={Once for All: Train One Network and Specialize it for Efficient Deployment},
  author={Han Cai and Chuang Gan and Tianzhe Wang and Zhekai Zhang and Song Han},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://arxiv.org/pdf/1908.09791.pdf}
}
```
