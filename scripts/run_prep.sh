#!/bin/bash

GPUNAME=$1
REPODIR=${2:-'/home/wimmerl/projects/wildlife-experiments/'}
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
--repo_dir=$REPODIR \
--root_dir=$ROOTDIR \
--img_dir=$IMGDIR && \
python scripts/prep_data.py --repo_dir=$REPODIR && \
python scripts/prep_experiments.py --repo_dir=$REPODIR
