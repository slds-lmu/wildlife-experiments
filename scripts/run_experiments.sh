#!/bin/bash

GPUNAME=$1
USERNAME=$2
EXPERIMENT=$3
if [[ -z $GPUNAME ]]
then
    echo `date`" - Missing mandatory arguments: GPU name. "
    exit 1
elif [[ $GPUNAME -eq 999 ]]
then
    export CUDA_VISIBLE_DEVICES=""
else
    export CUDA_VISIBLE_DEVICES=$GPUNAME
fi

python scripts/execute_experiments.py \
--repo_dir=/home/$USERNAME/projects/wildlife-experiments/ \
--experiment=$EXPERIMENT
