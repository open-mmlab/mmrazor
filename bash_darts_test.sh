#!/usr/bin/env sh


MKL_NUM_THREADS=4
OMP_NUM_THREADS=1

bash tools/slurm_test.sh mm_model spos_test configs/nas/darts/darts_subnet_1xb96_cifar10_2.0.py '/mnt/lustre/dongpeijie/darts_subnetnet_1xb96_cifar10_acc-97.32_20211222-e5727921_2.0.pth'
