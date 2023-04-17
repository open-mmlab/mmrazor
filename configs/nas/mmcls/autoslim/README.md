# AutoSlim

> [AutoSlim: Towards One-Shot Architecture Search for Channel Numbers](https://arxiv.org/abs/1903.11728)

<!-- [ALGORITHM] -->

## Abstract

We study how to set channel numbers in a neural network to achieve better accuracy under constrained resources (e.g., FLOPs, latency, memory footprint or model size). A simple and one-shot solution, named AutoSlim, is presented. Instead of training many network samples and searching with reinforcement learning, we train a single slimmable network to approximate the network accuracy of different channel configurations. We then iteratively evaluate the trained slimmable model and greedily slim the layer with minimal accuracy drop. By this single pass, we can obtain the optimized channel configurations under different resource constraints. We present experiments with MobileNet v1, MobileNet v2, ResNet-50 and RL-searched MNasNet on ImageNet classification. We show significant improvements over their default channel configurations. We also achieve better accuracy than recent channel pruning methods and neural architecture search methods.
Notably, by setting optimized channel numbers, our AutoSlim-MobileNet-v2 at 305M FLOPs achieves 74.2% top-1 accuracy, 2.4% better than default MobileNet-v2 (301M FLOPs), and even 0.2% better than RL-searched MNasNet (317M FLOPs). Our AutoSlim-ResNet-50 at 570M FLOPs, without depthwise convolutions, achieves 1.3% better accuracy than MobileNet-v1 (569M FLOPs).

![pipeline](https://user-images.githubusercontent.com/88702197/187425354-d90e4b36-e033-4dc0-b951-64a536e61b71.png)

## Get Started

### Supernet pre-training on ImageNet

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh \
  configs/nas/autoslim/autoslim_mbv2_1.5x_supernet_8xb256_in1k.py 4 \
  --work-dir $WORK_DIR
```

### Search for subnet on the trained supernet

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh \
  configs/nas/autoslim/autoslim_mbv2_1.5x_search_8xb256_in1k.py 4 \
  --work-dir $WORK_DIR --cfg-options load_from=$STEP1_CKPT
```

### Subnet retraining on ImageNet

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh \
  configs/nas/autoslim/autoslim_mbv2_subnet_8xb256_in1k.py 4 \
  --work-dir $WORK_DIR \
  --cfg-options algorithm.channel_cfg=configs/nas/autoslim/AUTOSLIM_MBV2_530M_OFFICIAL.yaml,configs/nas/autoslim/AUTOSLIM_MBV2_320M_OFFICIAL.yaml,configs/nas/autoslim/AUTOSLIM_MBV2_220M_OFFICIAL.yaml
```

### Split checkpoint

```bash
python ./tools/model_converters/split_checkpoint.py \
  configs/nas/autoslim/autoslim_mbv2_subnet_8xb256_in1k.py \
  $RETRAINED_CKPT \
  --channel-cfgs configs/nas/autoslim/AUTOSLIM_MBV2_530M_OFFICIAL.yaml configs/nas/autoslim/AUTOSLIM_MBV2_320M_OFFICIAL.yaml configs/nas/autoslim/AUTOSLIM_MBV2_220M_OFFICIAL.yaml
```

### Subnet inference

```bash
CUDA_VISIBLE_DEVICES=0 PORT=29500 ./tools/dist_test.sh \
  configs/nas/autoslim/autoslim_mbv2_subnet_8xb256_in1k.py \
  $SEARCHED_CKPT 1 --work-dir $WORK_DIR \
  --cfg-options algorithm.channel_cfg=configs/nas/autoslim/AUTOSLIM_MBV2_530M_OFFICIAL.yaml  # or modify the config directly
```

## Results and models

### Subnet retrain

| Supernet           | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                     Config                      |                                                                                                                                                                  Download                                                                                                                                                                   |                                                                                               Subnet                                                                                               |        Remark        |
| :----------------- | :-------: | -------: | :-------: | :-------: | :---------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------: |
| MobileNet v2(x1.5) |    6.5    |     0.53 |   74.23   |   91.74   | [config](./autoslim_mbv2_subnet_8xb256_in1k.py) | [model](https://download.openmmlab.com/mmrazor/v1/autoslim/autoslim_mbv2_subnet_8xb256_in1k_flops-530M_acc-74.23_20220715-aa8754fe.pth) \| [log](https://download.openmmlab.com/mmrazor/v0.1/pruning/autoslim/autoslim_mbv2_subnet_8xb256_in1k/autoslim_mbv2_subnet_8xb256_in1kautoslim_mbv2_subnet_8xb256_in1k_paper_channel_cfg.log.json) | [channel](https://download.openmmlab.com/mmrazor/v0.1/pruning/autoslim/autoslim_mbv2_subnet_8xb256_in1k/autoslim_mbv2_subnet_8xb256_in1k_flops-0.53M_acc-74.23_20211222-e5208bbd_channel_cfg.yaml) | official channel cfg |
| MobileNet v2(x1.5) |   5.77    |     0.32 |   72.73   |   90.83   | [config](./autoslim_mbv2_subnet_8xb256_in1k.py) | [model](https://download.openmmlab.com/mmrazor/v1/autoslim/autoslim_mbv2_subnet_8xb256_in1k_flops-320M_acc-72.73_20220715-9aa8f8ae.pth) \| [log](https://download.openmmlab.com/mmrazor/v0.1/pruning/autoslim/autoslim_mbv2_subnet_8xb256_in1k/autoslim_mbv2_subnet_8xb256_in1kautoslim_mbv2_subnet_8xb256_in1k_paper_channel_cfg.log.json) | [channel](https://download.openmmlab.com/mmrazor/v0.1/pruning/autoslim/autoslim_mbv2_subnet_8xb256_in1k/autoslim_mbv2_subnet_8xb256_in1k_flops-0.32M_acc-72.73_20211222-b5b0b33c_channel_cfg.yaml) | official channel cfg |
| MobileNet v2(x1.5) |   4.13    |     0.22 |   71.39   |   90.08   | [config](./autoslim_mbv2_subnet_8xb256_in1k.py) | [model](https://download.openmmlab.com/mmrazor/v1/autoslim/autoslim_mbv2_subnet_8xb256_in1k_flops-220M_acc-71.4_20220715-9c288f3b.pth) \| [log](https://download.openmmlab.com/mmrazor/v0.1/pruning/autoslim/autoslim_mbv2_subnet_8xb256_in1k/autoslim_mbv2_subnet_8xb256_in1kautoslim_mbv2_subnet_8xb256_in1k_paper_channel_cfg.log.json)  | [channel](https://download.openmmlab.com/mmrazor/v0.1/pruning/autoslim/autoslim_mbv2_subnet_8xb256_in1k/autoslim_mbv2_subnet_8xb256_in1k_flops-0.22M_acc-71.39_20211222-43117c7b_channel_cfg.yaml) | official channel cfg |

Note that we ran the official code and the Top-1 Acc of the models with official
channel cfg are 73.8%, 72.5% and 71.1%. And there are 3 differences between our
implementation and the official one.

1. The implementation of Label Smooth is slightly different.
2. Lighting is not used in our data pipeline. (Lighting is a kind of data
   augmentation which adjust images lighting using AlexNet-style PCA jitter.)
3. We do not recalibrating BN statistics after training.

## Citation

```latex
@article{yu2019autoslim,
  title={Autoslim: Towards one-shot architecture search for channel numbers},
  author={Yu, Jiahui and Huang, Thomas},
  journal={arXiv preprint arXiv:1903.11728},
  year={2019}
}
```
