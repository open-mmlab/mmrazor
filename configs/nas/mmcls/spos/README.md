# SPOS

> [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420)

<!-- [ALGORITHM] -->

## Abstract

We revisit the one-shot Neural Architecture Search (NAS) paradigm and analyze its advantages over existing NAS approaches. Existing one-shot method, however, is hard to train and not yet effective on large scale datasets like ImageNet. This work propose a Single Path One-Shot model to address the challenge in the training. Our central idea is to construct a simplified supernet, where all architectures are single paths so that weight co-adaption problem is alleviated. Training is performed by uniform path sampling. All architectures (and their weights) are trained fully and equally.
Comprehensive experiments verify that our approach is flexible and effective. It is easy to train and fast to search. It effortlessly supports complex search spaces (e.g., building blocks, channel, mixed-precision quantization) and different search constraints (e.g., FLOPs, latency). It is thus convenient to use for various needs. It achieves start-of-the-art performance on the large dataset ImageNet.

![pipeline](https://user-images.githubusercontent.com/88702197/187424862-c2f3fde1-4a48-4eda-9ff7-c65971b683ba.jpg)

## Get Started

### Step 1: Supernet pre-training on ImageNet

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh \
  configs/nas/spos/spos_supernet_shufflenetv2_8xb128_in1k.py 4 \
  --work-dir $WORK_DIR \
```

### Step 2: Search for subnet on the trained supernet

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh \
  configs/nas/spos/spos_evolution_search_shufflenetv2_8xb2048_in1k.py 4 \
  --work-dir $WORK_DIR --cfg-options load_from=$STEP1_CKPT
```

### Step 3: Subnet retraining on ImageNet

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh \
  configs/nas/spos/spos_subnet_shufflenetv2_8xb128_in1k.py 4 \
  --work-dir $WORK_DIR \
  --cfg-options model.init_cfg.checkpoint=$STEP2_CKPT

```

## Step 4: Subnet inference on ImageNet

```bash
CUDA_VISIBLE_DEVICES=0 PORT=29500 ./tools/dist_test.sh \
  configs/nas/spos/spos_subnet_shufflenetv2_8xb128_in1k.py \
  none 1 --work-dir $WORK_DIR \
  --cfg-options model.init_cfg.checkpoint=$STEP3_CKPT
```

## Results and models

| Dataset  |        Supernet        |                                                                            Subnet                                                                            | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                                           Config                                                            | Download                                                                                                                                                                                                                                                                                                                              |                            Remarks                            |
| :------: | :--------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------: | :------: | :-------: | :-------: | :-------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-----------------------------------------------------------: |
| ImageNet |      ShuffleNetV2      |  [mutable](https://download.openmmlab.com/mmrazor/v1/spos/spos_shufflenetv2_subnet_8xb128_in1k_flops_0.33M_acc_73.87_20220715-aa94d5ef_subnet_cfg_v3.yaml)   |   3.35    |   0.33   |   73.87   |   91.6    | [config](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/configs/nas/mmcls/spos/spos_subnet_shufflenetv2_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmrazor/v1/spos/spos_shufflenetv2_subnet_8xb128_in1k_flops_0.33M_acc_73.87_20211222-1f0a0b4d_v3.pth) \| [log](https://download.openmmlab.com/mmrazor/v0.1/nas/spos/spos_shufflenetv2_subnet_8xb128_in1k/spos_shufflenetv2_subnet_8xb128_in1k_flops_0.33M_acc_73.87_20211222-1f0a0b4d.log.json) |                       MMRazor searched                        |
| ImageNet | MobileNet-ProxylessGPU | [mutable](https://download.openmmlab.com/mmrazor/v0.1/nas/spos/spos_mobilenet_subnet/spos_angelnas_flops_0.49G_acc_75.98_20220307-54f4698f_mutable_cfg.yaml) |   5.94    |  0.49\*  |   75.98   |   92.77   |  [config](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/configs/nas/mmcls/spos/spos_mobilenet_subnet_8xb128_in1k.py)   |                                                                                                                                                                                                                                                                                                                                       | [AngleNAS](https://github.com/megvii-model/AngleNAS) searched |

**Note**:

1. There **might be(not all the case)** some small differences in our experiment in order to be consistent with other repos in OpenMMLab. For example,
   normalize images in data preprocessing; resize by cv2 rather than PIL in training; dropout is not used in network. **Please refer to corresponding config for details.**
2. For *ShuffleNetV2*, we retrain the subnet reported in paper with their official code, Top-1 is 73.6 and Top-5 is 91.6.
3. For *AngleNAS searched MobileNet-ProxylessGPU*, we obtain params and FLOPs using [this script](/tools/misc/get_flops.py), which may be different from [AngleNAS](https://github.com/megvii-model/AngleNAS#searched-models-with-abs).

## Citation

```latex
@inproceedings{guo2020single,
  title={Single path one-shot neural architecture search with uniform sampling},
  author={Guo, Zichao and Zhang, Xiangyu and Mu, Haoyuan and Heng, Wen and Liu, Zechun and Wei, Yichen and Sun, Jian},
  booktitle={European Conference on Computer Vision},
  pages={544--560},
  year={2020},
  organization={Springer}
}
```
