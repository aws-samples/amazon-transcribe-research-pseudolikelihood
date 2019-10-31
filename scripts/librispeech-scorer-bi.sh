#!/usr/bin/env bash

set -e
set -x

source_dir=$1
start=$2
end=$3
gpus=$4

model=bert-base-en-uncased
model_suffix=-380k
model_weights=params/bert-base-en-uncased-380k.params 

for x in `seq -w ${start} ${end}`
do
    target_dir=exps/librispeech-distill/${model}${model_suffix}
    mkdir -p ${target_dir}
    lpl score \
        --mode ref \
        --model ${model} \
        --gpus ${gpus} \
        --split-size 48 \
        --weights ${model_weights} \
        ${1}/part.${x} \
        > ${target_dir}/part.${x}.ref.score \
        2> >(tee ${target_dir}/part.${x}.ref.log >&2)
done
