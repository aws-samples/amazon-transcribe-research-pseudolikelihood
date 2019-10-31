## Pseudolikelihood Reranking with Masked Language Models

This repository will host code and experiments for the paper "Pseudolikelihood Reranking with Masked Language Models" by Julian Salazar, Davis Liang, Toan Q. Nguyen, and Katrin Kirchhoff, presented at the Workshop on Deep Learning for Low-Resource NLP (DeepLo), EMNLP-IJCNLP 2019.

**THIS CODEBASE IS A WORK IN PROGRESS.** It is licensed under the MIT-0 License. See the LICENSE file.

## Installation

The ideal setup is a fresh Deep Learning Base AMI (Ubuntu 18.04, v20) on a GPU instance (p3.2xlarge or better). **If and only if** you are in this environment, feel free to use `./dlami-setup.sh`, which will install the relevant packages and set up a [pipenv](https://docs.pipenv.org/en/latest/) environment for the current folder.

Otherwise, you can install this package directly. Look at `./dlami-setup.sh` for guidance if you run into issues.
```
pipenv install --python 3.7 --dev -e .
```
Note:
- We use constructs from Python 3.7.
- We assume CUDA 10 and MKL, so our Pipfile uses `mxnet-cu100mkl`. Replace with `mxnet-cu92`, `mxnet-mkl` (CPU only) as needed.
- If using pipenv, remember to activate the environment (`pipenv shell`).

## Example

Experiments outputs will be placed in `exps/`.  Refer to the ASR section for some examples with GPUs, pretrained weights, etc.

**NOTE:** We are awaiting permission from Shin et al., "Efficient Sentence Scoring with Bidirectional Language Models for Speech Recognition", 2019, to include their 100-best lists for LibriSpeech. If you wish, please contact them directly. They should placed in
```
data/librispeech-espnet/{dev-clean,dev-other,test-clean,test-other}.json
```

For now, we show the expected format in `data/example.json`.

First, we get scores from BERT Base (uncased):
```bash
mkdir -p exps/lpl/bert-base-en-uncased/
lpl score \
    --mode hyp \
    --model bert-base-en-uncased \
    --max-utts 1 \
    --split-size 100 \
    data/librispeech-espnet/dev-other.json \
    > exps/lpl/bert-base-en-uncased/dev-other-100.lm.json
```

We then rescore the acoustic model outputs:
```bash
set -e
for weight in $(seq 0 0.05 1.0) ; do
    echo ${weight}; \
    lpl rescore \
        --model bert-base-en-uncased \
        --weight ${weight} \
        data/librispeech-espnet/dev-other.json \
        exps/lpl/bert-base-en-uncased/dev-other-100.lm.json \
        > exps/lpl/bert-base-en-uncased/dev-other-100.lambda-${weight}.json
done
```

## Experiments

### ASR

Then for example, on a 4 GPU, Tesla V100 machine (`p3.8xlarge`):
```bash
set -e

### SCORING ###

# Stock BERT/RoBERTa base
for set in dev-clean dev-other test-clean test-other ; do
    for model in bert-base-en-uncased bert-base-en-cased roberta-base-en-cased ; do
        mkdir -p exps/lpl/${model}/
        echo ${set} ${model}
        lpl score \
            --mode hyp \
            --model ${model} \
            --gpus 0,1,2,3 \
            --split-size 2000 \
            data/librispeech-espnet/${set}.json \
            > exps/lpl/${model}/${set}.lm.json
    done
done

# Trained BERT base
for set in dev-clean dev-other test-clean test-other ; do
    for model in bert-base-en-uncased ; do
        mkdir -p exps/lpl/${model}-380k/
        echo ${set} ${model}-380k
        lpl score \
            --mode hyp \
            --model ${model} \
            --gpus 0,1,2,3 \
            --weights params/bert-base-en-uncased-380k.params \
            data/librispeech-espnet/${set}.json \
            > exps/lpl/${model}-380k/${set}.lm.json
    done
done

### RESCORING (DEV) ###

# Stock BERT/RoBERTa
for set in dev-clean ; do
    for model in bert-base-en-uncased bert-base-en-cased bert-large-en-uncased bert-large-en-cased roberta-base-en-cased roberta-large-en-cased ; do
        for weight in $(seq 0 0.05 1.0) ; do
            echo ${set} ${model} ${weight}; \
            lpl rescore \
                --model ${model} \
                --weight ${weight} \
                data/librispeech-espnet/${set}.json \
                exps/lpl/${model}/${set}.lm.json \
                > exps/lpl/${model}/${set}.lambda-${weight}.json
        done
    done
done

### RESCORING (TEST) ###

# Once you have your hyperparameter, evaluate test
for set in test-clean ; do
    for tup in bert-base-en-uncased,,0.35 bert-base-en-cased,,0.35 bert-large-en-uncased,,0.40 bert-large-en-cased,,0.35 ; do
        IFS="," read model suffix weight <<< "${tup}"
        echo ${set} ${model}${suffix} ${weight}
        lpl rescore \
            --model ${model} \
            --weight ${weight} \
            data/librispeech-espnet/${set}.json \
            exps/lpl/${model}${suffix}/${set}.lm.json \
            > exps/lpl/${model}${suffix}/${set}.lambda-${weight}.json
        done
    done
done

```

### NMT

**TODO**

## Development

- To run unit tests and coverage, run `pytest --cov=src/lpl` in the root directory.
- To run static typing checks, run `mypy --strict src/lpl` in the root directory.
