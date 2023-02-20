#!/bin/bash

GPUNAME=$1
PERSONALFOLDER=$2
ROOTDIR=${3:-'/common/bothmannl/'}
IMGDIR=${4:-'wildlife_images/usecase2/original_images/'}

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

python scripts/prep_config.py \
--repo_dir=$PERSONALFOLDER/wildlife-experiments/ \
--root_dir=$ROOTDIR \
--img_dir=$IMGDIR
