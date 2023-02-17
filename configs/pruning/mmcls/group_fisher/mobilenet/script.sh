# act mode
bash ./tools/dist_train.sh configs/pruning/mmcls/group_fisher/mobilenet/group_fisher_act_prune_mobilenet-v2_8xb32_in1k.py 8
bash ./tools/dist_train.sh configs/pruning/mmcls/group_fisher/mobilenet/group_fisher_act_finetune_mobilenet-v2_8xb32_in1k.py 8

# flops mode
bash ./tools/dist_train.sh configs/pruning/mmcls/group_fisher/mobilenet/group_fisher_flops_prune_mobilenet-v2_8xb32_in1k.py 8
bash ./tools/dist_train.sh configs/pruning/mmcls/group_fisher/mobilenet/group_fisher_flops_finetune_mobilenet-v2_8xb32_in1k.py 8
