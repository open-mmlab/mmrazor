#!/bin/bash

#SBATCH --gpus=1

module load cuda/11.3

source activate openmmlab

export FULL_TEST='true'
export PYTHONUNBUFFERED=1
export DEBUG='true'
# export MP=4
# export TEST_DATA='true'

python -m pytest tests -s -k 'test_prune_tracer_model'