#!/usr/bin/env sh


MKL_NUM_THREADS=4
OMP_NUM_THREADS=1

# DetNAS train
# srun --partition=mm_model \
#     --job-name=detnas_train \
#     --gres=gpu:8 \
#     --ntasks=8 \
#     --ntasks-per-node=8 \
#     --cpus-per-task=8 \
#     --kill-on-bad-exit=1 \
#     python tools/train.py configs/nas/detnas/detnas_supernet_shufflenetv2_coco_1x_2.0_frcnn.py

# bash tools/slurm_train.sh mm_model detnas_train configs/nas/detnas/detnas_supernet_shufflenetv2_coco_1x_2.0_frcnn.py /mnt/lustre/dongpeijie/checkpoints/tests/detnas_pretrain_test


# bash tools/slurm_test.sh mm_model detnas_test configs/nas/detnas/detnas_supernet_shufflenetv2_coco_1x_2.0_frcnn.py /mnt/lustre/dongpeijie/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f_2.0.pth

# DetNAS test
srun --partition=mm_model \
    --job-name=detnas_test \
    --gres=gpu:1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --kill-on-bad-exit=1 \
    --quotatype=auto \
    python tools/test.py configs/nas/detnas/detnas_subnet_shufflenetv2_8xb128_in1k_2.0_frcnn.py "/mnt/lustre/dongpeijie/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f_2.0.pth" --launcher=slurm
