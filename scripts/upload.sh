#! /bin/sh

rsync -auv \
--exclude .git \
--exclude .venv \
--exclude .data \
--exclude .idea \
--exclude .mypy_cache \
--exclude results_* \
--exclude models \
--exclude wandb \
--exclude logs \
--max-size=50m \
. \
wimmerl@gpuserver.stat.uni-muenchen.de:projects/wildlife-experiments
