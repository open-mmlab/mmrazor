# act mode
bash ./tools/dist_train.sh configs/pruning/mmcls/group_fisher/resnet50/group_fisher_act_prune_resnet50_8xb32_in1k.py.py 8
bash ./tools/dist_train.sh configs/pruning/mmcls/group_fisher/resnet50/group_fisher_act_finetune_resnet50_8xb32_in1k.py.py 8

# flops mode
bash ./tools/dist_train.sh configs/pruning/mmcls/group_fisher/resnet50/group_fisher_flops_prune_resnet50_8xb32_in1k.py.py 8
bash ./tools/dist_train.sh configs/pruning/mmcls/group_fisher/resnet50/group_fisher_flops_finetune_resnet50_8xb32_in1k.py 8
