# python ./tools/train.py ./projects/group_fisher/configs/mmcls/vgg/vgg_group_fisher_prune.py
# python ./tools/train.py ./projects/group_fisher/configs/mmcls/vgg/vgg_group_fisher_finetune.py

python ./tools/train.py ./projects/group_fisher/configs/mmcls/vgg/vgg_group_fisher_prune_flop.py
python ./tools/train.py ./projects/group_fisher/configs/mmcls/vgg/vgg_group_fisher_finetune_flop.py
