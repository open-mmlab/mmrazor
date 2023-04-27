_base_ = './group-fisher-finetune_retinanet_resnet50_8xb2_coco.py'

pruned_path = './work_dirs/group-fisher-pruning_retinanet_resnet50_8xb2_coco_act/flops_0.50.pth'  # noqa

model = dict(
    algorithm=dict(init_cfg=dict(type='Pretrained',
                                 checkpoint=pruned_path), ), )
