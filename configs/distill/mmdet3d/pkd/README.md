# PKD

> [PKD: General Distillation Framework for Object Detectors via Pearson Correlation Coefficient](https://arxiv.org/abs/2207.02039)

<!-- [ALGORITHM] -->

## Abstract

Knowledge distillation(KD) is a widely-used technique to train compact models in object detection. However, there is still a lack of study on how to distill between heterogeneous detectors. In this paper, we empirically find that better FPN features from a heterogeneous teacher detector can help the student although their detection heads and label assignments are different. However, directly aligning the feature maps to distill detectors suffers from two problems. First, the difference in feature magnitude between the teacher and the student could enforce overly strict constraints on the student. Second, the FPN stages and channels with large feature magnitude from the teacher model could dominate the gradient of distillation loss, which will overwhelm the effects of other features in KD and introduce much noise. To address the above issues, we propose to imitate features with Pearson Correlation Coefficient to focus on the relational information from the teacher and relax constraints on the magnitude of the features. Our method consistently outperforms the existing detection KD methods and works for both homogeneous and heterogeneous student-teacher pairs. Furthermore, it converges faster. With a powerful MaskRCNN-Swin detector as the teacher, ResNet-50 based RetinaNet and FCOS achieve 41.5% and 43.9% mAP on COCO2017, which are 4.1% and 4.8% higher than the baseline, respectively.

![pipeline](https://user-images.githubusercontent.com/88702197/187424502-d8efb7a3-c40c-4e53-a36c-bd947de464a4.png)

## Results and models

### Detection

| Location |  Dataset   |                                                                         Teacher                                                                         |     Student      | Lr schd | mAP  | mAP(T) | mAP(S) |                             Config                             | Download                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| :------: | :--------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------: | :-----: | :--: | :----: | :----: | :------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|   FPN    | nus-mono3d | [FCOS3d-R101](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py) | [FCOS3d-R50](<>) |   1x    | 29.3 |  32.1  |  26.8  | [config](pkd_fpn_fcos3d_r101_fcos3d_r50_8xb2-1x_nus-mono3d.py) | [teacher](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth) \|[model](https://download.openmmlab.com/mmrazor/v1/pkd/pkd_fcos3d_w10/pkd_fpn_fcos3d_r101_fcos3d_r50_8xb2-1x_nus-mono3d_20220928_234557-0b51b62e.pth?versionId=CAEQThiBgMC8sdC0oBgiIDAwOWE2OWUyNDU1NTQ1MjBhZTY1NmNjODZmMDZkZTM2) \| [log](https://download.openmmlab.com/mmrazor/v1/pkd/pkd_fcos3d_w10/pkd_fpn_fcos3d_r101_fcos3d_r50_8xb2-1x_nus-mono3d_20220928_234557-0b51b62e.json?versionId=CAEQThiBgIDrvdC0oBgiIDNmNGNkNDZhM2RmNjQ1MmI4ZDM0OGNmYmFkYjk5ZjFi) |

## Citation

```latex
@article{cao2022pkd,
  title={PKD: General Distillation Framework for Object Detectors via Pearson Correlation Coefficient},
  author={Cao, Weihan and Zhang, Yifan and Gao, Jianfei and Cheng, Anda and Cheng, Ke and Cheng, Jian},
  journal={arXiv preprint arXiv:2207.02039},
  year={2022}
}
```
