#!/bin/bash

GPUNAME=$1
if [[ -z $GPUNAME ]]
then
    echo `date`" - Missing mandatory arguments: GPU name. "
    exit 1
elif [[ $GPUNAME -eq 99 ]]
then
    export CUDA_VISIBLE_DEVICES=""
else
    export CUDA_VISIBLE_DEVICES=$GPUNAME
fi

python scripts/prep_config.py && \
python scripts/prep_data.py && \
python scripts/prep_experiments.py
