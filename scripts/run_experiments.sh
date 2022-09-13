#!/bin/bash

GPUNAME=$1
EXPERIMENT=$2
REPODIR=${3:-'/home/wimmerl/projects/wildlife-experiments/'}
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
--repo_dir=$REPODIR \
--experiment=$EXPERIMENT
