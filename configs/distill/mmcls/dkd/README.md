# Decoupled Knowledge Distillation

> [Decoupled Knowledge Distillation](https://arxiv.org/pdf/2203.08679.pdf)

<!-- [ALGORITHM] -->

## Abstract

State-of-the-art distillation methods are mainly based on distilling deep features from intermediate layers, while the significance of logit distillation is greatly overlooked. To provide a novel viewpoint to study logit distillation, we reformulate the classical KD loss into two parts, i.e., target class knowledge distillation (TCKD) and non-target class knowledge distillation (NCKD). We empirically investigate and prove the effects of the two parts: TCKD transfers knowledge concerning the "difficulty" of training samples, while NCKD is the prominent reason why logit distillation works. More importantly, we reveal that the classical KD loss is a coupled formulation, which (1) suppresses the effectiveness of NCKD and (2) limits the flexibility to balance these two parts. To address these issues, we present Decoupled Knowledge Distillation (DKD), enabling TCKD and NCKD to play their roles more efficiently and flexibly. Compared with complex feature-based methods, our DKD achieves comparable or even better results and has better training efficiency on CIFAR-100, ImageNet, and MS-COCO datasets for image classification and object detection tasks. This paper proves the great potential of logit distillation, and we hope it will be helpful for future research. The code is available at this \[https://github.com/megvii-research/mdistiller\]

![avatar](../../../../docs/en/imgs/model_zoo/dkd/dkd.png)

## Results and models

### Classification

| Dataset  | Model     | Teacher   | Top-1 (%) | Top-5 (%) | Configs                                                                                           | Download                                                                                             |
| -------- | --------- | --------- | --------- | --------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| ImageNet | ResNet-18 | ResNet-34 | 71.368    | 90.256    | [config](dkd_logits_r34_r18_8xb32_in1k.py) | [model & log](https://autolink.sensetime.com/pages/model/share/afc68955-e25d-4488-b044-5e801b3ff62f) |

## Citation

```latex
@article{zhao2022decoupled,
  title={Decoupled Knowledge Distillation},
  author={Zhao, Borui and Cui, Quan and Song, Renjie and Qiu, Yiyu and Liang, Jiajun},
  journal={arXiv preprint arXiv:2203.08679},
  year={2022}
}
```

## Getting Started

Download teacher ckpt from

https://mmclassification.readthedocs.io/en/latest/papers/resnet.html

### Distillation training.

```bash
sh tools/slurm_train.sh $PARTITION $JOB_NAME \
  configs/distill/mmcls/dkd/dkd_logits_r34_r18_8xb32_in1k.py \
  $DISTILLATION_WORK_DIR
```

## Acknowledgement

Shout out to Davidgzx for his special contribution.
