# AutoFormer

> [Searching Transformers for Visual Recognition](https://arxiv.org/abs/2107.00651)

<!-- [ALGORITHM] -->

## Abstract

Recently, pure transformer-based models have shown
great potentials for vision tasks such as image classification and detection. However, the design of transformer networks is challenging. It has been observed that the depth,
embedding dimension, and number of heads can largely affect the performance of vision transformers. Previous models configure these dimensions based upon manual crafting. In this work, we propose a new one-shot architecture
search framework, namely AutoFormer, dedicated to vision
transformer search. AutoFormer entangles the weights of
different blocks in the same layers during supernet training. Benefiting from the strategy, the trained supernet allows thousands of subnets to be very well-trained. Specifically, the performance of these subnets with weights inherited from the supernet is comparable to those retrained
from scratch. Besides, the searched models, which we refer to AutoFormers, surpass the recent state-of-the-arts such
as ViT and DeiT. In particular, AutoFormer-tiny/small/base
achieve 74.7%/81.7%/82.4% top-1 accuracy on ImageNet
with 5.7M/22.9M/53.7M parameters, respectively. Lastly,
we verify the transferability of AutoFormer by providing
the performance on downstream benchmarks and distillation experiments.

![pipeline](/docs/en/imgs/model_zoo/autoformer/pipeline.png)

## Introduction

### Supernet pre-training on ImageNet

```bash
python ./tools/train.py \
  configs/nas/mmcls/autoformer/autoformer_supernet_32xb256_in1k.py \
  --work-dir $WORK_DIR
```

### Search for subnet on the trained supernet

```bash
sh tools/train.py \
  configs/nas/mmcls/autoformer/autoformer_search_8xb128_in1k.py \
  $STEP1_CKPT \
  --work-dir $WORK_DIR
```

## Results and models

| Dataset  | Supernet |                                                                                                                                               Subnet                                                                                                                                                | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                     Config                      | Download                                                                                                                                                                    |     Remarks      |
| :------: | :------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------: | :------: | :-------: | :-------: | :---------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------: |
| ImageNet |   vit    | [mutable](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v0.1/nas/spos/spos_shufflenetv2_subnet_8xb128_in1k/spos_shufflenetv2_subnet_8xb128_in1k_flops_0.33M_acc_73.87_20211222-454627be_mutable_cfg.yaml?versionId=CAEQHxiBgICw5b6I7xciIGY5MjVmNWFhY2U5MjQzN2M4NDViYzI2YWRmYWE1YzQx) |  52.472   |   10.2   |   82.48   |   95.99   | [config](./autoformer_supernet_32xb256_in1k.py) | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/x.pth) \| [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v0.1/nas/spos/x.log.json) | MMRazor searched |

**Note**:

1. There are some small differences in our experiment in order to be consistent with mmrazor repo. For example, we set the max value of embed_channels 624 while the original repo set it 640. However, the original repo only search 528, 576, 624 embed_channels, so set 624 can also get the same result with orifinal paper.
2. The original paper get 82.4 top-1 acc with 53.7M Params while we get 82.48 top-1 acc with 52.47M Params.

## Citation

```latex
@article{xu2021autoformer,
  title={Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting},
  author={Xu, Jiehui and Wang, Jianmin and Long, Mingsheng and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

Footer
