GPUNAME=$1
if [[ -z $GPUNAME ]];
then
    echo `date`" - Missing mandatory arguments: GPU name. "
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$1GPUNAME

SCRIPT=$2
if [[ -z $SCRIPT ]];
then
    echo `date`" - Missing mandatory arguments: script to run. "
    exit 1
fi

python scripts/$2SCRIPT
