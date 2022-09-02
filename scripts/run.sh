#!/bin/bash

GPUNAME=$1
if [[ -z $GPUNAME ]]
then
    echo `date`" - Missing mandatory arguments: GPU name. "
    exit 1
elif [[ $1 -eq 99 ]]
then
    export CUDA_VISIBLE_DEVICES=""
else
    export CUDA_VISIBLE_DEVICES=$1
fi
SCRIPT=$2
if [[ -z $SCRIPT ]]
then
    echo `date`" - Missing mandatory arguments: script to run. "
    exit 1
fi

python scripts/$2.py
