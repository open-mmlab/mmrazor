# CWD

> [Channel-wise Knowledge Distillation for Dense Prediction](https://arxiv.org/abs/2011.13256)

<!-- [ALGORITHM] -->

## Abstract

Knowledge distillation (KD) has been proven to be a simple and effective tool for training compact models. Almost all KD variants for dense prediction tasks align the student and teacher networks' feature maps in the spatial domain, typically by minimizing point-wise and/or pair-wise discrepancy. Observing that in semantic segmentation, some layers' feature activations of each channel tend to encode saliency of scene categories (analogue to class activation mapping), we propose to align features channel-wise between the student and teacher networks. To this end, we first transform the feature map of each channel into a probability map using softmax normalization, and then minimize the Kullback-Leibler (KL) divergence of the corresponding channels of the two networks. By doing so, our method focuses on mimicking the soft distributions of channels between networks. In particular, the KL divergence enables learning to pay more attention to the most salient regions of the channel-wise maps, presumably corresponding to the most useful signals for semantic segmentation. Experiments demonstrate that our channel-wise distillation outperforms almost all existing spatial distillation methods for semantic segmentation considerably, and requires less computational cost during training. We consistently achieve superior performance on three benchmarks with various network structures.

![pipeline](https://user-images.githubusercontent.com/88702197/187424502-d8efb7a3-c40c-4e53-a36c-bd947de464a4.png)

## Results and models

### Segmentation

| Location |  Dataset   |                                                             Teacher                                                              |                                                            Student                                                             | mIoU  | mIoU(T) | mIou(S) |    Config    | Download                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| :------: | :--------: | :------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :---: | :-----: | :-----: | :----------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  logits  | cityscapes | [pspnet_r101](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes.py) | [pspnet_r18](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/pspnet/pspnet_r18-d8_512x1024_80k_cityscapes.py) | 75.54 |  79.76  |  74.87  | [config](<>) | [teacher](https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth) \|[model](https://download.openmmlab.com/mmrazor/v0.1/distill/cwd/cwd_cls_head_pspnet_r101_d8_pspnet_r18_d8_512x1024_cityscapes_80k/cwd_cls_head_pspnet_r101_d8_pspnet_r18_d8_512x1024_cityscapes_80k_mIoU-75.54_20211222-3a26ee1c.pth) \| [log](https://download.openmmlab.com/mmrazor/v0.1/distill/cwd/cwd_cls_head_pspnet_r101_d8_pspnet_r18_d8_512x1024_cityscapes_80k/cwd_cls_head_pspnet_r101_d8_pspnet_r18_d8_512x1024_cityscapes_80k_20211212_205711.log.json) |

### Detection

| Location | Dataset |                                                     Teacher                                                      |                                                Student                                                 | mAP  | mAP(T) | mAP(S) |    Config    | Download                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| :------: | :-----: | :--------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :--: | :----: | :----: | :----------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| cls head |  COCO   | [gfl_r101_2x](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py) | [gfl_r50_1x](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl/gfl_r50_fpn_1x_coco.py) | 41.9 |  44.7  |  40.2  | [config](<>) | [teacher](https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth) \|[model](https://download.openmmlab.com/mmrazor/v0.1/distill/cwd/cwd_cls_head_gfl_r101_fpn_gfl_r50_fpn_1x_coco/cwd_cls_head_gfl_r101_fpn_gfl_r50_fpn_1x_coco_20211222-655dff39.pth) \| [log](https://download.openmmlab.com/mmrazor/v0.1/distill/cwd/cwd_cls_head_gfl_r101_fpn_gfl_r50_fpn_1x_coco/cwd_cls_head_gfl_r101_fpn_gfl_r50_fpn_1x_coco_20211212_205444.log.json) |

## Citation

```latex
@inproceedings{shu2021channel,
  title={Channel-Wise Knowledge Distillation for Dense Prediction},
  author={Shu, Changyong and Liu, Yifan and Gao, Jianfei and Yan, Zheng and Shen, Chunhua},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5311--5320},
  year={2021}
}
```
