#!/bin/bash

PERSONALFOLDER=$1
SEED=$2

python scripts/prep_data.py \
--repo_dir=$PERSONALFOLDER/wildlife-experiments/ \
--random_seed=$SEED
