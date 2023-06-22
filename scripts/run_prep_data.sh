#!/bin/bash

SEED=$1
PERSONALFOLDER=${2:-'/home/wimmerl/projects'}

python scripts/prep_data.py \
--repo_dir=$PERSONALFOLDER/wildlife-experiments/ \
--random_seed=$SEED
