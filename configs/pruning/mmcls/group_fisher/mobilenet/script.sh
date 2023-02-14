bash ./tools/dist_train.sh ./projects/group_fisher/configs/mmcls/mobilenet/mobilenet_v2_group_fisher_prune.py 8
bash ./tools/dist_train.sh ./projects/group_fisher/configs/mmcls/mobilenet/mobilenet_v2_group_fisher_finetune.py 8

bash ./tools/dist_train.sh projects/group_fisher/configs/mmcls/mobilenet/flop/mobilenet_v2_group_fisher_prune_flop.py 8
bash ./tools/dist_train.sh projects/group_fisher/configs/mmcls/mobilenet/flop/mobilenet_v2_group_fisher_finetune_flop.py 8
