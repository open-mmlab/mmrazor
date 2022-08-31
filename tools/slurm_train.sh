#!/usr/bin/env bash

set -x

PARTITION="mm_model"
JOB_NAME="search_config"
CONFIG="configs/nas/mmcls/spos/spos_shufflenet_search_8xb128_in1k.py"
CKPT="/mnt/lustre/humu/v2/experiments/train_demo/epoch_117.pth"
WORK_DIR="../experiments/search_config"
GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --quotatype=auto \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --cfg-options dist.port=12345 load_from=${CKPT} --launcher="slurm" ${PY_ARGS}
