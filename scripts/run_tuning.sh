#!/bin/bash

GPUNAME=$1
PERSONALFOLDER=$2
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

python scripts/execute_tuning.py \
--repo_dir=$PERSONALFOLDER/wildlife-experiments/