#!/usr/bin/env sh


MKL_NUM_THREADS=4
OMP_NUM_THREADS=1

# train
# srun --partition=mm_model \
#     --job-name=spos_train \
#     --gres=gpu:8 \
#     --ntasks=8 \
#     --ntasks-per-node=8 \
#     --cpus-per-task=8 \
#     --kill-on-bad-exit=1 \
#     python tools/train.py configs/nas/spos/spos_supernet_shufflenetv2_8xb128_in1k_2.0_example.py

# bash tools/slurm_train.sh mm_model spos_train configs/nas/spos/spos_supernet_shufflenetv2_8xb128_in1k_2.0_example.py /mnt/lustre/dongpeijie/checkpoints/work_dirs/spos_format_output

# bash tools/slurm_train.sh mm_model spos_retrain configs/nas/spos/spos_subnet_shufflenetv2_8xb128_in1k_2.0_example.py /mnt/lustre/dongpeijie/checkpoints/work_dirs/spos_retrain_detnas_with_ceph

# 55% wrong settings of PolyLR
# bash tools/slurm_train.sh mm_model spos_retrain_w_cj configs/nas/spos/spos_subnet_shufflenetv2_8xb128_in1k_2.0_example.py /mnt/lustre/dongpeijie/checkpoints/work_dirs/spos_retrain_detnas_with_ceph

# fix setting of PolyLR and rerun with colorjittor
# bash tools/slurm_train.sh mm_model spos_retrain_w_cj configs/nas/spos/spos_subnet_shufflenetv2_8xb128_in1k_2.0_example.py /mnt/lustre/dongpeijie/checkpoints/work_dirs/retrain_detnas_spos_with_colorjittor

# fix setting of PolyLR and rerun w/o colorjittor
# bash tools/slurm_train.sh mm_model spos_retrain_wo_cj configs/nas/spos/spos_subnet_shufflenetv2_8xb128_in1k_2.0_example_wo_colorjittor.py /mnt/lustre/dongpeijie/checkpoints/work_dirs/retrain_detnas_spos_wo_colorjittor

# fix setting of optimizer decay[wo cj] (paramwise_cfg)
# bash tools/slurm_train.sh mm_model spos_retrain_fix_decay_wo_cj configs/nas/spos/spos_subnet_shufflenetv2_8xb128_in1k_2.0_example_wo_colorjittor.py /mnt/lustre/dongpeijie/checkpoints/work_dirs/retrain_detnas_spos_retrain_fix_decay_wo_cj

# fix setting of optimizer decay[with cj] (paramwise_cfg)
# bash tools/slurm_train.sh mm_model spos_retrain_fix_decay_w_cj configs/nas/spos/spos_subnet_shufflenetv2_8xb128_in1k_2.0_example.py /mnt/lustre/dongpeijie/checkpoints/work_dirs/retrain_detnas_spos_retrain_fix_decay_w_cj



# SPOS test
# srun --partition=mm_model \
#     --job-name=spos_test \
#     --gres=gpu:1 \
#     --ntasks=1 \
#     --ntasks-per-node=1 \
#     --cpus-per-task=8 \
#     --kill-on-bad-exit=1 \
#     python tools/test.py configs/nas/spos/spos_subnet_shufflenetv2_8xb128_in1k_2.0_example.py "/mnt/lustre/dongpeijie/spos_shufflenetv2_subnet_8xb128_in1k_flops_0.33M_acc_73.87_20211222-1f0a0b4d_2.0.pth"


bash tools/slurm_test.sh mm_model spos_test configs/nas/spos/spos_subnet_shufflenetv2_8xb128_in1k_2.0_example.py '/mnt/lustre/dongpeijie/detnas_subnet_shufflenetv2_8xb128_in1k_acc-74.08_20211223-92e9b66a_2.0.pth'

# bash tools/slurm_train.sh mm_model spos_retrain configs/nas/spos/spos_subnet_shufflenetv2_8xb128_in1k_2.0_example.py /mnt/lustre/dongpeijie/checkpoints/work_dirs/spos_retrain_detnas_spos
