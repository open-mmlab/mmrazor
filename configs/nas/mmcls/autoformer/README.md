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

## Get Started

### Step 1: Supernet pre-training on ImageNet

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh \
  configs/nas/mmcls/autoformer/autoformer_supernet_32xb256_in1k.py 4 \
  --work-dir $WORK_DIR
```

### Step 2: Search for subnet on the trained supernet

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh \
  configs/nas/mmcls/autoformer/autoformer_search_8xb128_in1k.py 4 \
  --work-dir $WORK_DIR --cfg-options load_from=$STEP1_CKPT
```

### Step 3: Subnet inference on ImageNet

```bash
CUDA_VISIBLE_DEVICES=0 PORT=29500 ./tools/dist_test.sh \
  configs/nas/mmcls/autoformer/autoformer_subnet_8xb128_in1k.py \
  none 1 --work-dir $WORK_DIR \
  --cfg-options model.init_cfg.checkpoint=$STEP1_CKPT model.init_weight_from_supernet=True

```

## Results and models

| Dataset  | Supernet |                               Subnet                               | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                     Config                      | Download                                                                                                                                                                                                                                                  |     Remarks      |
| :------: | :------: | :----------------------------------------------------------------: | :-------: | :------: | :-------: | :-------: | :---------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------: |
| ImageNet |   vit    | [mutable](./configs/nas/mmcls/autoformer/AUTOFORMER_SUBNET_B.yaml) |  54.319   |  10.57   |   82.47   |   95.99   | [config](./autoformer_supernet_32xb256_in1k.py) | [model](https://download.openmmlab.com/mmrazor/v1/autoformer/autoformer_supernet_32xb256_in1k_20220919_110144-c658ce8f.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/autoformer/autoformer_supernet_32xb256_in1k_20220919_110144-c658ce8f.json) | MMRazor searched |

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
