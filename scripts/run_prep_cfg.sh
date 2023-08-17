#!/bin/bash

GPUNAME=$1
PERSONALFOLDER=${2:-'/home/wimmerl/projects'}
IMGDIR=${3:-'/common/bothmannl/wildlife_images/usecase2/original_images/'}

if [[ -z $GPUNAME ]]
then
    echo "Missing mandatory arguments: GPU name. "
    exit 1
elif [[ $GPUNAME -eq 999 ]]
then
    export CUDA_VISIBLE_DEVICES=""
else
    export CUDA_VISIBLE_DEVICES=$GPUNAME
fi

python scripts/prep_config.py \
--repo_dir="$PERSONALFOLDER"/wildlife-experiments/ \
--img_dir="$IMGDIR"
