#! /bin/sh

USERNAME=$1

if [[ -z $USERNAME ]]
then
    echo `date`" - Missing mandatory arguments: user name for GPU server. "
    exit 1
fi

rsync -auv \
--exclude .git \
--exclude .json \
--exclude .venv \
--exclude .data \
--exclude .idea \
--exclude .mypy_cache \
--exclude results_* \
--exclude models \
--exclude logs \
--max-size=50m \
. \
$USERNAME@gpuserver.stat.uni-muenchen.de:projects/wildlife-experiments
