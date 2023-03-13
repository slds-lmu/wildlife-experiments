#!/bin/bash

GPUNAME=$1
PERSONALFOLDER=$2
EXPERIMENT=$3
SEED=$4
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
--repo_dir=$PERSONALFOLDER/wildlife-experiments/ \
--experiment=$EXPERIMENT
--random_seed=$SEED
