# Pseudolikelihood Reranking with Masked Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This package uses masked LMs like BERT, RoBERTa, and XLM to [score sentences](#scoring) and [rerank n-best lists](#reranking) via log-pseudolikelihood scores. [Experiments](#examples) in automatic speech recognition (ASR) and neural machine translation (NMT) are included. Work in progress.

**Reference:** Julian Salazar, Davis Liang, Toan Q. Nguyen, Katrin Kirchhoff. "[Pseudolikelihood Reranking with Masked Language Models](https://arxiv.org/abs/1910.14659)", *Workshop on Deep Learning for Low-Resource NLP (DeepLo)*, 2019. 

## Installation

Our environment is a fresh GPU instance (p3.2xlarge or better) with a Deep Learning Base AMI (Ubuntu 18.04, v20.2). *If and only if you are in this environment*, you can safely use `./dlami-setup.sh`, which sets the CUDA version, installs relevant packages, and creates a [pipenv](https://pipenv.kennethreitz.org/en/latest/) environment in the current folder.

Otherwise, install this package directly. **If using `pipenv`:**
```bash
pipenv install --dev -e .
```
**If using `pip`:** You must install the entries under `Pipfile > [packages]` manually:
```bash
pip install -e .
pip install gluonnlp==0.8.1 regex ... # etc.
```
**Note:**
- Python 3.7+ is required.
- CUDA 10.1 and Intel MKL are assumed; our Pipfile requires `mxnet-cu101mkl`. Uninstall and replace with `mxnet-cu92` (CUDA 9.2, non-Intel CPU), `mxnet-mkl` (Intel CPU only), etc. as needed.

## Usage

### Scoring

There are three scoring modes, depending on the model:
- Log-pseudolikelihood (LPL) score: BERT, RoBERTa, multilingual BERT, XLM
- Log probability score: GPT-2
- Maskless LPL score: see [LibriSpeech maskless finetuning](examples/asr-librispeech-espnet/README.md)

Run `lpl score --help` to see supported models, etc. See `examples/demo/format.json` for the file format. For inputs, "score" is optional. Outputs will add "score" fields containing LPL scores.

We score hypotheses for 3 utterances of LibriSpeech `dev-other` on CPU using BERT base (uncased):
```bash
lpl score \
    --mode hyp \
    --model bert-base-en-uncased \
    --max-utts 3 \
    --gpus -1 \
    --split-size 100 \
    examples/asr-librispeech-espnet/data/dev-other.am.json \
    > examples/demo/dev-other-3.lm.json
```

When scoring more sentences you may want to use GPUs; see [Examples](#examples).

### Reranking

One can rescore n-best lists via log-linear interpolation. Run `lpl rescore --help` to see all options. Input one is a file with original scores; input two are scores from `lpl score`.

We rescore acoustic scores (from `dev-other.am.json`) using BERT's scores (from previous section), under different LM weights:
```bash
for weight in $(seq 0 0.05 1.0) ; do
    echo "lambda=${weight}"; \
    lpl rescore \
        --model bert-base-en-uncased \
        --weight ${weight} \
        examples/asr-librispeech-espnet/data/dev-other.am.json \
        examples/demo/dev-other-3.lm.json \
        > examples/demo/dev-other-3.lambda-${weight}.json
done
```

### Maskless finetuning

**TODO**

## Examples

The following correspond to experiments from the paper:
- [Speech Recognition > LibriSpeech > ESPnet (encoder-decoder)](examples/asr-librispeech-espnet/README.md)
- Machine Translation  (**TODO**)

## Development

- To run unit tests and coverage, run `pytest --cov=src/lpl` in the root directory.
- To run static typing checks, run `mypy --strict src/lpl` in the root directory.
