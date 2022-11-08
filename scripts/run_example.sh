#!/bin/bash

CONFIGFILE=$1
TASK=$2

if [[ -z $CONFIGFILE ]]
then
    echo `date`" - Missing mandatory argument: path to configuration file. "
    exit 1
fi
if [[ -z $TASK ]]
then
    echo `date`" - Missing mandatory argument: task to be executed. "
    exit 1
fi

python scripts/compute_example.py \
-cf=$CONFIGFILE \
-tk=$TASK
