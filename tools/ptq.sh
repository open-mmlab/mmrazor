#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29104}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/ptq.py $CONFIG --launcher pytorch ${@:3}
