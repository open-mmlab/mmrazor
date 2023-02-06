bash ./tools/dist_train.sh ./projects/group_fisher/configs/mmdet/group-fisher-pruning_retinanet_resnet50_8xb2_coco.py 8
bash ./tools/dist_train.sh ./projects/group_fisher/configs/mmdet/group-fisher-finetune_retinanet_resnet50_8xb2_coco.py 8


bash ./tools/dist_train.sh ./projects/group_fisher/configs/mmdet/group-fisher-pruning_retinanet_resnet50_8xb2_coco_act.py 8
bash ./tools/dist_train.sh ./projects/group_fisher/configs/mmdet/group-fisher-finetune_retinanet_resnet50_8xb2_coco_act.py 8
