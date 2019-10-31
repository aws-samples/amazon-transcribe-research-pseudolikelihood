#!/usr/bin/env bash

set -e
set -x

source_dir=$1
start=$2
end=$3
gpus=$4

model=gpt2-345m-en-cased

for x in `seq -w ${start} ${end}`
do
    target_dir=exps/librispeech-distill/${model}
    mkdir -p ${target_dir}
    lpl score \
        --mode ref \
        --model ${model} \
        --gpus ${gpus} \
        --split-size 32 \
        ${1}/part.${x} \
        > ${target_dir}/part.${x}.ref.score \
        2> >(tee ${target_dir}/part.${x}.ref.log >&2)
done
