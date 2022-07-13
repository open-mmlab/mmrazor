#!/usr/bin/env sh


MKL_NUM_THREADS=4
OMP_NUM_THREADS=1



# bash tools/slurm_train.sh mm_model detnas_train configs/nas/detnas/detnas_supernet_shufflenetv2_coco_1x_2.0_frcnn.py /mnt/lustre/dongpeijie/checkpoints/tests/detnas_pretrain_test


bash tools/slurm_test.sh mm_model angle_test configs/nas/spos/spos_subnet_mobilenet_proxyless_gpu_8xb128_in1k_2.0.py /mnt/lustre/dongpeijie/spos_angelnas_flops_0.49G_acc_75.98_20220307-54f4698f_2.0.pth
