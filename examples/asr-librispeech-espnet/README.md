## ASR (LibriSpeech)

We include the 100-best LibriSpeech decoding outputs from "[Effective Sentence Scoring Method using Bidirectional Language Model for Speech Recognition](https://arxiv.org/abs/1905.06655)" (Shin et al., 2019) with the authors' permission; please cite their work if reusing their lists. The outputs come from a 5-layer encoder, 1-layer decoder BLSTMP model implemented in ESPnet. The files are `data/*.am.json`.

The split sizes are per 16GB Tesla V100 GPUs; in our case (`p3.8xlarge`) there are four. Scale appropriately for your per-GPU memory.

**TODO: Model artifacts.**

### Scoring

```bash
# Stock BERT/RoBERTa base
for set in dev-clean dev-other test-clean test-other ; do
    for model in bert-base-en-uncased bert-base-en-cased roberta-base-en-cased ; do
        mkdir -p output/${model}/
        echo ${set} ${model}
        lpl score \
            --mode hyp \
            --model ${model} \
            --gpus 0,1,2,3 \
            --split-size 2000 \
            data/${set}.am.json \
            > output/${model}/${set}.lm.json
    done
done
# Trained BERT base
for set in dev-clean dev-other test-clean test-other ; do
    for model in bert-base-en-uncased ; do
        mkdir -p output/${model}-380k/
        echo ${set} ${model}-380k
        lpl score \
            --mode hyp \
            --model ${model} \
            --gpus 0,1,2,3 \
            --split-size 2000 \
            --weights params/bert-base-en-uncased-380k.params \
            data/${set}.am.json \
            > output/${model}-380k/${set}.lm.json
    done
done
```

### Reranking

```bash
# Stock BERT/RoBERTa on development set
for set in dev-clean ; do
    for model in bert-base-en-uncased bert-base-en-cased roberta-base-en-cased ; do
        for weight in $(seq 0 0.05 1.0) ; do
            echo ${set} ${model} ${weight}; \
            lpl rescore \
                --model ${model} \
                --weight ${weight} \
                data/${set}.am.json \
                output/${model}/${set}.lm.json \
                > output/${model}/${set}.lambda-${weight}.json
        done
    done
done
# Once you have the best hyperparameter, evaluate test
for set in test-clean ; do
    for tup in bert-base-en-uncased,,0.35 bert-base-en-cased,,0.35 ; do
        IFS="," read model suffix weight <<< "${tup}"
        echo ${set} ${model}${suffix} ${weight}
        lpl rescore \
            --model ${model} \
            --weight ${weight} \
            data/${set}.am.json \
            output/${model}${suffix}/${set}.lm.json \
            > output/${model}${suffix}/${set}.lambda-${weight}.json
        done
    done
done
```

### Binning

**TODO** To compute cross-entropy statistics:
```bash
lpl bin
```

### Maskless finetuning

**TODO** To train a regression model towards sentence scores:
```bash
lpl finetune
```