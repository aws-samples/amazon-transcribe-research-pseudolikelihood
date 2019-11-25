# Pseudolikelihood Reranking with Masked Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**WORK IN PROGRESS.** This code implements *pseudolikelihood reranking* using pretrained masked LMs on n-best lists, with specific experiments in automatic speech recognition (ASR) and neural machine translation (NMT).

Julian Salazar, Davis Liang, Toan Q. Nguyen, Katrin Kirchhoff. "[Pseudolikelihood Reranking with Masked Language Models](https://arxiv.org/abs/1910.14659)", *Workshop on Deep Learning for Low-Resource NLP (DeepLo)*, 2019. 

## Installation

Our setup is a fresh GPU instance (p3.2xlarge or better) with a Deep Learning Base AMI (Ubuntu 18.04, v20.2). *If and only if you are in this environment*, you can safely use `./dlami-setup.sh` to set the CUDA version, install relevant packages, and set up a [pipenv](https://pipenv.kennethreitz.org/en/latest/) environment in the current folder.

Otherwise, install this package directly. Look at `./dlami-setup.sh` for guidance if you run into issues.
```bash
pipenv install --dev -e .
```
Caveats:
- Python 3.7+ is required.
- CUDA 10.1 and Intel MKL are assumed; our Pipfile requires `mxnet-cu101mkl`. Replace with `mxnet-cu92`, `mxnet-mkl` (CPU only), etc. as needed.
- Use non-pipenv environments at your own risk.

## Example

Experiment outputs will be placed in `exps/`.  Refer to the ASR section for command line arguments with GPUs, pretrained weights, etc. Our commands are for rescoring the included ASR decoding outputs, but you can replace these with your own. The format is summarized in `data/example.json`.

To demonstrate, we score the first 3 utterances of LibriSpeech `dev-other` on CPU using BERT base (uncased):
```bash
mkdir -p exps/lpl/bert-base-en-uncased/
lpl score \
    --mode hyp \
    --model bert-base-en-uncased \
    --max-utts 3 \
    --gpus -1 \
    --split-size 100 \
    data/librispeech-espnet/dev-other.json \
    > exps/lpl/bert-base-en-uncased/dev-other-3.lm.json
```

We then rescore the acoustic model outputs:
```bash
set -e
for weight in $(seq 0 0.05 1.0) ; do
    echo "lambda=${weight}"; \
    lpl rescore \
        --model bert-base-en-uncased \
        --weight ${weight} \
        data/librispeech-espnet/dev-other.json \
        exps/lpl/bert-base-en-uncased/dev-other-3.lm.json \
        > exps/lpl/bert-base-en-uncased/dev-other-3.lambda-${weight}.json
done
```

## ASR (LibriSpeech)

We include the 100-best LibriSpeech decoding outputs from "[Effective Sentence Scoring Method using Bidirectional Language Model for Speech Recognition](https://arxiv.org/abs/1905.06655)" (Shin et al., 2019) with the authors' permission; please cite their work if reusing their lists. The outputs come from a 5-layer encoder, 1-layer decoder BLSTMP model implemented in ESPnet. The files are located in `data/librispeech-espnet/*.json`.

The split sizes below are for a 4 GPU, Tesla V100 machine (`p3.8xlarge`). Scale appropriately for your GPU memory.

### Scoring

```bash
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
```

### Reranking

```bash
# Stock BERT/RoBERTa on development set
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

# Once you have the best hyperparameter, evaluate test
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

### Binning

**TODO** To compute cross-entropy statistics:
```
lpl bin
```

### Maskless finetuning

**TODO** To train a regression model towards sentence scores:
```
lpl finetune
```

## NMT (TED Talks, IWSLT'15)

**TODO**

### Scoring

### Reranking

## Development

- To run unit tests and coverage, run `pytest --cov=src/lpl` in the root directory.
- To run static typing checks, run `mypy --strict src/lpl` in the root directory.
