# act mode
bash ./tools/dist_train.sh configs/pruning/mmdet/group_fisher/retinanet/group_fisher_act_prune_retinanet_r50_fpn_1x_coco.py 8
bash ./tools/dist_train.sh configs/pruning/mmdet/group_fisher/retinanet/group_fisher_act_finetune_retinanet_r50_fpn_1x_coco.py 8

# flops mode
bash ./tools/dist_train.sh configs/pruning/mmdet/group_fisher/retinanet/group_fisher_flops_prune_retinanet_r50_fpn_1x_coco.py 8
bash ./tools/dist_train.sh configs/pruning/mmdet/group_fisher/retinanet/group_fisher_flops_finetune_retinanet_r50_fpn_1x_coco.py 8
