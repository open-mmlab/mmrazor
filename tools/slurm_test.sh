#!/usr/bin/env bash

set -x

PARTITION="mm_model"
JOB_NAME="adaround"
WORK_DIR="../experiments/adaround"
CONFIG="configs/quantization/ptq/adaround.py"
CHECKPOINT="/mnt/petrelfs/humu/share/resnet18_8xb32_in1k_20210831-fbbb1da6.pth"
GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:5}
SRUN_ARGS=${SRUN_ARGS:-""}

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
    python -u tools/test.py ${CONFIG} ${CHECKPOINT} --work-dir=$WORK_DIR --launcher="slurm" ${PY_ARGS}
