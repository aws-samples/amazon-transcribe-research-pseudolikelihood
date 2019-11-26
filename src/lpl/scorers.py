from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gluonnlp as nlp
import gluonnlp.data.batchify as btf
from gluonnlp.model.bert import BERTModel, RoBERTaModel
import mxnet as mx
from mxnet.gluon import Block
from mxnet.gluon.data import SimpleDataset
import numpy as np

from .loaders import Corpus, ScoredCorpus


class BaseScorer(ABC):
    """A wrapper around a model which can score utterances
    """

    def __init__(self, model: Block, vocab: nlp.Vocab, tokenizer, ctxs: List[mx.Context], eos: Optional[bool] = None, capitalize: Optional[bool] = None) -> None:
        self._model = model
        self._vocab = vocab
        self._tokenizer = tokenizer
        self._ctxs = ctxs
        self._eos = eos
        self._capitalize = capitalize
        self._max_length = 1024


    def _apply_tokenizer_opts(self, sent: str) -> str:
        if self._eos:
            sent += '.'
        if self._capitalize:
            sent = sent.capitalize()
        return sent


    def _corpus_to_data(self, corpus, split_size, ratio, num_workers: int, shuffle: bool=False):

        # Turn corpus into a dataset
        dataset = self.corpus_to_dataset(corpus)

        # TODO: There is a 'by-design' bug in FixedBucketSampler with num_shards > 0, where it silently reuses the last utterances:
        # https://github.com/dmlc/gluon-nlp/blame/b1b61d3f90cf795c7b48b6d109db7b7b96fa21ff/src/gluonnlp/data/sampler.py#L398
        # batch_sampler = nlp.data.sampler.FixedBucketSampler([sent_tuple[2] for sent_tuple in dataset], batch_size=split_size, ratio=ratio, num_shards=len(self._ctxs), shuffle=False)
        # Hence, we use num_shards = 0 and do gluon's split_data
        batch_sampler = nlp.data.sampler.FixedBucketSampler([sent_tuple[2] for sent_tuple in dataset], batch_size=split_size, ratio=ratio, num_shards=0, shuffle=shuffle)

        logging.info(batch_sampler.stats())
        dataloader = nlp.data.ShardedDataLoader(dataset, pin_memory=True, batch_sampler=batch_sampler, batchify_fn=self._batchify_fn, num_workers=num_workers, thread_pool=True)

        return dataset, batch_sampler, dataloader


    def _true_tok_lens(self, dataset):

        # Compute sum (assumes dataset is in order; skips are allowed)
        prev_sent_idx = None
        true_tok_lens = []
        for tup in dataset:
            curr_sent_idx = tup[0]
            valid_length = tup[2]
            if curr_sent_idx != prev_sent_idx:
                prev_sent_idx = curr_sent_idx
                true_tok_lens.append(valid_length - 2)

        return true_tok_lens


    def _split_batch(self, batch):
        # TODO: Write tests about batching over multiple GPUs and getting the same scores
        # TODO: SEE COMMENT ABOVE REGARDING FIXEDBUCKETSAMPLER
        return zip(*[mx.gluon.utils.split_data(batch_compo, len(self._ctxs), batch_axis=0, even_split=False) for batch_compo in batch])


    def score(self, corpus: Corpus, temp: float = 1.0, split_size: int = 2000, ratio: float = 1, num_workers: int = 10, per_token: bool = False) -> List[float]:

        assert not per_token

        ctx_cpu = mx.Context('cpu')

        # Get MXNet data objects
        dataset, batch_sampler, dataloader = self._corpus_to_data(corpus, split_size, ratio, num_workers)

        # Get number of tokens
        true_tok_lens = self._true_tok_lens(dataset)

        # Compute scores
        scores = np.zeros((len(corpus),)) 
        scores_per_ctx = [mx.nd.zeros((len(corpus),), ctx=ctx) for ctx in self._ctxs]

        sent_count = 0
        batch_log_interval = 20

        batch_score_accumulation = 1
        batch_sent_idxs_per_ctx = [[] for ctx in self._ctxs]
        batch_scores_per_ctx = [[] for ctx in self._ctxs]

        def sum_accumulated_scores():
            for ctx_idx in range(len(self._ctxs)):
                for batch_sent_idxs, batch_scores in zip(batch_sent_idxs_per_ctx[ctx_idx], batch_scores_per_ctx[ctx_idx]):
                    np.add.at(scores, batch_sent_idxs.asnumpy(), batch_scores.asnumpy())
                batch_sent_idxs_per_ctx[ctx_idx] = []
                batch_scores_per_ctx[ctx_idx] = []

        # For now just predicts the first non-cls token
        for batch_id, batch in enumerate(dataloader):

            batch = self._split_batch(batch)

            batch_size = self._batch_ops(batch, batch_sent_idxs_per_ctx, batch_scores_per_ctx, temp)

            # Ideally we'd accumulate the scores when possible, but something like the below won't work
            # > scores[sent_idxs] += out
            # See In[21] in https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html.
            # Hence, aggregation is done synchronously, every so often
            # (though batch_score_accumulation = 1 seems best, since bucketing is effective in reducing GPU disparity)
            if len(batch_sent_idxs_per_ctx[0]) == batch_score_accumulation:   
                sum_accumulated_scores()

            # Progress
            sent_count += batch_size
            if (batch_id+1) % batch_log_interval == 0:
                logging.info("{} sents of {}, batch {} of {}".format(sent_count, len(dataset), batch_id+1, len(batch_sampler)))

        # TODO: Test score accumulation
        # In case there are leftovers
        sum_accumulated_scores()

        return scores.tolist(), true_tok_lens


class LLScorer(BaseScorer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Turn Dataset into Dataloader
        # (GPT-2 does not have a padding token; but the padding token shouldn't matter anyways)
        self._batchify_fn = btf.Tuple(btf.Stack(dtype='int32'), btf.Pad(pad_val=np.iinfo(np.int32).max, dtype='int32'),
                              btf.Stack(dtype='int32'))


    def corpus_to_dataset(self, corpus: Corpus) -> SimpleDataset:

        sents_expanded = []

        for sent_idx, sent in enumerate(corpus.values()):
            sent = self._apply_tokenizer_opts(sent)
            tokens_original = [self._vocab.eos_token] + self._tokenizer(sent) + [self._vocab.eos_token]
            ids_original = np.array(self._tokenizer.convert_tokens_to_ids(tokens_original))

            # Enforce max length
            if len(ids_original) > self._max_length:
                logging.error("Line #{} is too long; will output score of 0 and omit in token counts (but not yet in word counts!)".format(sent_idx+1))
            else:
                sents_expanded += [(sent_idx, ids_original, len(ids_original))]

        return SimpleDataset(sents_expanded)


    def _batch_ops(self, batch, batch_sent_idxs_per_ctx, batch_scores_per_ctx, temp) -> int:

        batch_size = 0

        for ctx_idx, (sent_idxs, token_ids, valid_length) in enumerate(batch):

            ctx = self._ctxs[ctx_idx]
            batch_size += sent_idxs.shape[0]
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)

            # out is ((batch size, max_seq_len, vocab size), new states)
            out = self._model(token_ids)

            # Get the probability computed for the correct token
            split_size = token_ids.shape[0]

            # TODO: Manual numerically-stable softmax
            # https://stackoverflow.com/questions/42599498/numercially-stable-softmax
            # Because we only need one scalar
            out = out[0].log_softmax(temperature=temp)

            # Get scores ignoring our version of CLS (the 'G.') and SEP (the '<|endoftext|>' after the terminal 'G.')
            # Recall that the softmaxes here are for predicting the >>next<< token
            batch_sent_idxs_per_ctx[ctx_idx].append(sent_idxs)

            out_final = mx.nd.zeros((out.shape[0],), ctx=ctx)
            for i in range(out.shape[0]):
                # Get scores ignoring our version of CLS (the 'G.') and SEP (the '<|endoftext|>' after the terminal 'G.')
                # Recall that the softmaxes here are for predicting the >>next<< token
                out_final[i] = out[i, list(range(valid_length[i].asscalar()-2)), token_ids[i, 1:(valid_length[i].asscalar()-1)]].sum()
                # out = out[list(range(split_size)), valid_length-2, token_ids[i, 1:(valid_length[i]-1)]]
            batch_scores_per_ctx[ctx_idx].append(out_final)

        return batch_size


class LLBinner(LLScorer):

    # TODO: WIP
    def _bin_ops(self, batch, bin_counts_per_ctx, bin_sums_per_ctx, temp) -> int:

        batch_size = 0

        for ctx_idx, (sent_idxs, token_ids, valid_length) in enumerate(batch):

            ctx = self._ctxs[ctx_idx]
            batch_size += sent_idxs.shape[0]
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)

            # out is ((batch size, max_seq_len, vocab size), new states)
            out = self._model(token_ids)
            out = out[0].log_softmax(temperature=temp)

            ### <DIFFERENT>            
            for i in range(out.shape[0]):
                # Get scores ignoring our version of CLS and SEP (the '<|endoftext|>' after the terminal 'G.')
                # Recall that the softmaxes here are for predicting the >>next<< token
                # -2 since we don't care about the prediction after the punctuation
                num_bins = valid_length[i].asscalar()-2
                bin_counts_per_ctx[ctx_idx][num_bins,:num_bins] += 1
                bin_sums_per_ctx[ctx_idx][num_bins,:num_bins] += out[i, list(range(num_bins)), token_ids[i, 1:(num_bins+1)]]
            ### </DIFFERENT>

            # Not separated by length:
            # ### <DIFFERENT>            
            # for i in range(out.shape[0]):
            #     # Get scores ignoring our version of CLS and SEP (the '<|endoftext|>' after the terminal 'G.')
            #     # Recall that the softmaxes here are for predicting the >>next<< token
            #     # -2 since we don't care about the prediction after the punctuation
            #     num_bins = valid_length[i].asscalar()-2
            #     bin_counts_per_ctx[ctx_idx][:num_bins] += 1
            #     bin_sums_per_ctx[ctx_idx][:num_bins] += out[i, list(range(num_bins)), token_ids[i, 1:(num_bins+1)]]
            # ### </DIFFERENT>

        return batch_size


    def bin(self, corpus: Corpus, temp: float = 1.0, split_size: int = 2000, ratio: float = 1, num_workers: int = 10) -> List[float]:

        ctx_cpu = mx.Context('cpu')

        # Get MXNet data objects
        dataset, batch_sampler, dataloader = self._corpus_to_data(corpus, split_size, ratio, num_workers)

        # Get number of tokens
        true_tok_lens = self._true_tok_lens(dataset)

        ### <DIFFERENT>
        max_length = 256
        # Compute bins
        # First axis is sentence length
        bin_counts = np.zeros((max_length, max_length))
        bin_counts_per_ctx = [mx.nd.zeros((max_length, max_length), ctx=ctx) for ctx in self._ctxs]
        bin_sums = np.zeros((max_length, max_length))
        bin_sums_per_ctx = [mx.nd.zeros((max_length, max_length), ctx=ctx) for ctx in self._ctxs]
        ### </DIFFERENT>

        # Not separated by length:
        # ### <DIFFERENT>
        # # Compute bins
        # bin_counts = np.zeros((self._max_length,))
        # bin_counts_per_ctx = [mx.nd.zeros((self._max_length,), ctx=ctx) for ctx in self._ctxs]
        # bin_sums = np.zeros((self._max_length,))
        # bin_sums_per_ctx = [mx.nd.zeros((self._max_length,), ctx=ctx) for ctx in self._ctxs]
        # ### </DIFFERENT>

        sent_count = 0
        batch_log_interval = 20

        # For now just predicts the first non-cls token
        for batch_id, batch in enumerate(dataloader):

            batch = self._split_batch(batch)

            batch_size = self._bin_ops(batch, bin_counts_per_ctx, bin_sums_per_ctx, temp)

            # Progress
            sent_count += batch_size
            if (batch_id+1) % batch_log_interval == 0:
                logging.info("{} sents of {}, batch {} of {}".format(sent_count, len(dataset), batch_id+1, len(batch_sampler)))

        # Accumulate the counts
        for ctx_idx in range(len(self._ctxs)):
            bin_counts += bin_counts_per_ctx[ctx_idx].asnumpy()
            bin_sums += bin_sums_per_ctx[ctx_idx].asnumpy()

        return bin_counts, bin_sums


# TODO: Dedup with BaseScorer's score()
class LPLScorer(BaseScorer):
    """For models that need every token to be masked
    """


    def __init__(self, *args, **kwargs):
        self._wwm = kwargs.pop('wwm') if 'wwm' in kwargs else False
        super().__init__(*args, **kwargs)


    def _ids_to_masked(self, token_ids: np.ndarray) -> List[Tuple[np.ndarray, List[int]]]:

        # Here:
        # token_ids = [2 ... ... 1012 3], where 2=[CLS], (optionally) 1012='.', 3=[SEP]

        token_ids_masked_list = []

        assert (not self._wwm)

        mask_indices = []
        if self._wwm:
            for idx, token_id in enumerate(token_ids):
                if self._tokenizer.is_first_subword(self._vocab.idx_to_token[token_id]):
                    mask_indices.append([idx])
                else:
                    mask_indices[-1].append(idx)
        else:
            mask_indices = [[mask_pos] for mask_pos in range(len(token_ids))]

        # # We don't mask the [CLS], [SEP] for now for PLL
        mask_indices = mask_indices[1:-1]

        mask_token_id = self._vocab.token_to_idx[self._vocab.mask_token]
        for mask_set in mask_indices:
            token_ids_masked = token_ids.copy()
            token_ids_masked[mask_set] = mask_token_id

            # TODO: Do a less-wasteful approach for WWM
            if self._wwm:
                # We add multiple entries with whole word masked
                # perform (duplicated) inference on each, then extract
                # and then perform averaging in the batch logic below

                # # Sum
                # for i in range(len(mask_set)):
                #     sents_expanded.append((sent_idx, token_ids_masked, valid_length, [mask_set[i]], token_ids[mask_set[i]], 1))

                # Average
                for i in range(len(mask_set)):
                    sents_expanded.append((sent_idx, token_ids_masked, valid_length, [mask_set[i]], token_ids[mask_set[i]], len(mask_set)))

            else:
                token_ids_masked_list.append((token_ids_masked, mask_set))

        return token_ids_masked_list


    def print_record(self, record):
        readable_sent = [self._vocab.idx_to_token[tid] for tid in record[1]]
        logging.info("""
sent_idx = {},
text = {},
all toks = {},
masked_id = {}
        """.format(record[0], readable_sent, record[2], record[3], record[4]))


    def corpus_to_dataset(self, corpus: Corpus) -> SimpleDataset:

        sents_expanded = []

        for sent_idx, sent in enumerate(corpus.values()):
            sent = self._apply_tokenizer_opts(sent)
            tokens_original = [self._vocab.cls_token] + self._tokenizer(sent) + [self._vocab.sep_token]
            ids_original = np.array(self._tokenizer.convert_tokens_to_ids(tokens_original))

            # Enforce max length
            if len(ids_original) > self._max_length:
                logging.error("Line #{} is too long; will output score of 0 and omit in token counts (but not yet in word counts!)".format(sent_idx+1))
            else:
                ids_masked = self._ids_to_masked(ids_original)
                sents_expanded += [(
                        sent_idx,
                        ids,
                        len(ids_original),
                        mask_set,
                        ids_original[mask_set],
                        1)
                    for ids, mask_set in ids_masked]

        return SimpleDataset(sents_expanded)


    def score(self, corpus: Corpus, temp: float = 1.0, split_size: int = 2000, ratio: float = 0, num_workers: int = 10, per_token: bool = False) -> List[float]:

        ctx_cpu = mx.Context('cpu')

        # Turn corpus into a BERT-ready Dataset
        dataset = self.corpus_to_dataset(corpus)

        # Turn Dataset into Dataloader
        batchify_fn = btf.Tuple(btf.Stack(dtype='int32'), btf.Pad(pad_val=self._vocab.token_to_idx[self._vocab.padding_token], dtype='int32'),
                              btf.Stack(dtype='float32'), btf.Stack(dtype='float32'),
                              btf.Stack(dtype='int32'), btf.Stack(dtype='float32'))

        # TODO: There is a 'by-design' bug in FixedBucketSampler with num_shards > 0, where it silently reuses the last utterances:
        # https://github.com/dmlc/gluon-nlp/blame/b1b61d3f90cf795c7b48b6d109db7b7b96fa21ff/src/gluonnlp/data/sampler.py#L398
        # batch_sampler = nlp.data.sampler.FixedBucketSampler([sent_tuple[2] for sent_tuple in dataset], batch_size=split_size, ratio=ratio, num_shards=len(self._ctxs), shuffle=False)
        # Hence, we use num_shards = 0 and do gluon's split_data
        batch_sampler = nlp.data.sampler.FixedBucketSampler([sent_tuple[2] for sent_tuple in dataset], batch_size=split_size, ratio=ratio, num_shards=0, shuffle=False)

        logging.info(batch_sampler.stats())
        dataloader = nlp.data.ShardedDataLoader(dataset, pin_memory=True, batch_sampler=batch_sampler, batchify_fn=batchify_fn, num_workers=num_workers, thread_pool=True)

        # Get lengths in tokens (assumes dataset is in order)
        prev_sent_idx = None
        true_tok_lens = []
        for (curr_sent_idx, _, valid_length, _, _, _) in dataset:
            if curr_sent_idx != prev_sent_idx:
                prev_sent_idx = curr_sent_idx
                true_tok_lens.append(valid_length - 2)

        # Compute scores (total or per-position)
        if per_token:
            scores_per_token = [[None]*(true_tok_len+2) for true_tok_len in true_tok_lens]
        else:
            scores = np.zeros((len(corpus),))

        sent_count = 0
        batch_log_interval = 20

        batch_score_accumulation = 1
        batch_sent_idxs_per_ctx = [[] for ctx in self._ctxs]
        batch_scores_per_ctx = [[] for ctx in self._ctxs]
        batch_masked_positions_per_ctx = [[] for ctx in self._ctxs]

        def sum_accumulated_scores():
            for ctx_idx in range(len(self._ctxs)):
                for batch_sent_idxs, batch_scores, batch_masked_positions in zip(batch_sent_idxs_per_ctx[ctx_idx], batch_scores_per_ctx[ctx_idx], batch_masked_positions_per_ctx[ctx_idx]):
                    if per_token:
                        # Slow; only use when necessary
                        for batch_sent_idx, batch_score, batch_masked_position in zip(batch_sent_idxs, batch_scores, batch_masked_positions):
                            scores_per_token[batch_sent_idx.asscalar()][int(batch_masked_position.asscalar())] = batch_score.asscalar().item()
                    else:
                        np.add.at(scores, batch_sent_idxs.asnumpy(), batch_scores.asnumpy())
                batch_sent_idxs_per_ctx[ctx_idx] = []
                batch_scores_per_ctx[ctx_idx] = []
                batch_masked_positions_per_ctx[ctx_idx] = []

        # For now just predicts the first non-cls token
        for batch_id, batch in enumerate(dataloader):

            batch_size = 0

            # TODO: Write tests about batching over multiple GPUs and getting the same scores
            # TODO: SEE COMMENT ABOVE REGARDING FIXEDBUCKETSAMPLER
            batch = zip(*[mx.gluon.utils.split_data(batch_compo, len(self._ctxs), batch_axis=0, even_split=False) for batch_compo in batch])

            for ctx_idx, (sent_idxs, token_ids, valid_length, masked_positions, token_masked_ids, normalization) in enumerate(batch):

                ctx = self._ctxs[ctx_idx]
                batch_size += sent_idxs.shape[0]
                token_ids = token_ids.as_in_context(ctx)
                valid_length = valid_length.as_in_context(ctx)
                masked_positions = masked_positions.as_in_context(ctx).reshape(-1, 1)

                if isinstance(self._model, RoBERTaModel):
                    out = self._model(token_ids, valid_length, masked_positions)
                else:
                    segment_ids = mx.nd.zeros(shape=token_ids.shape, ctx=ctx)
                    out = self._model(token_ids, segment_ids, valid_length, masked_positions)

                # Get the probability computed for the correct token
                split_size = token_ids.shape[0]
                # out[0] contains the representations
                # out[1] is what contains the distribution for the masked

                # TODO: Manual numerically-stable softmax
                # https://stackoverflow.com/questions/42599498/numercially-stable-softmax
                # Because we only need one scalar
                out = out[1].log_softmax(temperature=temp)

                # Save the scores at the masked indices
                batch_sent_idxs_per_ctx[ctx_idx].append(sent_idxs)
                out = out[list(range(split_size)), [0]*split_size, token_masked_ids.as_in_context(ctx).reshape(-1)]
                batch_scores_per_ctx[ctx_idx].append(out)
                batch_masked_positions_per_ctx[ctx_idx].append(masked_positions)

            # Ideally we'd accumulate the scores when possible, but something like the below won't work
            # > scores[sent_idxs] += out
            # See In[21] in https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html.
            # Hence, aggregation is done synchronously, every so often
            # (though batch_score_accumulation = 1 seems best, since bucketing is effective in reducing GPU disparity)
            if len(batch_sent_idxs_per_ctx[0]) == batch_score_accumulation:   
                sum_accumulated_scores()

            # Progress
            sent_count += batch_size
            if (batch_id+1) % batch_log_interval == 0:
                logging.info("{} sents of {}, batch {} of {}".format(sent_count, len(dataset), batch_id+1, len(batch_sampler)))

        # TODO: Test score accumulation
        # In case there are leftovers
        sum_accumulated_scores()

        if per_token:
            return scores_per_token, true_tok_lens
        else:
            return scores.tolist(), true_tok_lens


class LPLBinner(LPLScorer):

    # TODO: WIP
    def bin(self, corpus: Corpus, temp: float = 1.0, split_size: int = 2000, ratio: float = 0, num_workers: int = 10) -> List[float]:

        ctx_cpu = mx.Context('cpu')

        # Turn corpus into a BERT-ready Dataset
        dataset = self.corpus_to_dataset(corpus)

        # Turn Dataset into Dataloader
        batchify_fn = btf.Tuple(btf.Stack(dtype='int32'), btf.Pad(pad_val=self._vocab.token_to_idx[self._vocab.padding_token], dtype='int32'),
                              btf.Stack(dtype='float32'), btf.Stack(dtype='int32'),
                              btf.Stack(dtype='int32'), btf.Stack(dtype='float32'))

        # TODO: There is a 'by-design' bug in FixedBucketSampler with num_shards > 0, where it silently reuses the last utterances:
        # https://github.com/dmlc/gluon-nlp/blame/b1b61d3f90cf795c7b48b6d109db7b7b96fa21ff/src/gluonnlp/data/sampler.py#L398
        # batch_sampler = nlp.data.sampler.FixedBucketSampler([sent_tuple[2] for sent_tuple in dataset], batch_size=split_size, ratio=ratio, num_shards=len(self._ctxs), shuffle=False)
        # Hence, we use num_shards = 0 and do gluon's split_data
        batch_sampler = nlp.data.sampler.FixedBucketSampler([sent_tuple[2] for sent_tuple in dataset], batch_size=split_size, ratio=ratio, num_shards=0, shuffle=False)

        logging.info(batch_sampler.stats())
        dataloader = nlp.data.ShardedDataLoader(dataset, pin_memory=True, batch_sampler=batch_sampler, batchify_fn=batchify_fn, num_workers=num_workers, thread_pool=True)

        ### <DIFFERENT>
        max_length = 256
        # Compute bins
        # First axis is sentence length
        bin_counts = np.zeros((max_length, max_length))
        bin_counts_per_ctx = [mx.nd.zeros((max_length, max_length), ctx=ctx) for ctx in self._ctxs]
        bin_sums = np.zeros((max_length, max_length))
        bin_sums_per_ctx = [mx.nd.zeros((max_length, max_length), ctx=ctx) for ctx in self._ctxs]
        ### </DIFFERENT>

        # Compute sum (assumes dataset is in order)
        prev_sent_idx = None
        true_tok_lens = []
        for (curr_sent_idx, _, valid_length, _, _, _) in dataset:
            if curr_sent_idx != prev_sent_idx:
                prev_sent_idx = curr_sent_idx
                true_tok_lens.append(valid_length - 2)

        sent_count = 0
        batch_log_interval = 20

        # For now just predicts the first non-cls token
        for batch_id, batch in enumerate(dataloader):

            batch_size = 0

            # TODO: Write tests about batching over multiple GPUs and getting the same scores
            # TODO: SEE COMMENT ABOVE REGARDING FIXEDBUCKETSAMPLER
            batch = zip(*[mx.gluon.utils.split_data(batch_compo, len(self._ctxs), batch_axis=0, even_split=False) for batch_compo in batch])

            for ctx_idx, (sent_idxs, token_ids, valid_length, masked_positions, token_masked_ids, normalization) in enumerate(batch):

                ctx = self._ctxs[ctx_idx]
                batch_size += sent_idxs.shape[0]
                token_ids = token_ids.as_in_context(ctx)
                valid_length = valid_length.as_in_context(ctx)
                segment_ids = mx.nd.zeros(shape=token_ids.shape, ctx=ctx)
                masked_positions = masked_positions.as_in_context(ctx).reshape(-1, 1)
                out = self._model(token_ids, segment_ids, valid_length, masked_positions)

                # Get the probability computed for the correct token
                split_size = token_ids.shape[0]
                # out[0] contains the representations
                # out[1] is what contains the distribution for the masked
                out = out[1].log_softmax(temperature=temp)

                ### <DIFFERENT>
                token_masked_ids = token_masked_ids.as_in_context(ctx).reshape(-1)
                for i in range(out.shape[0]):
                    num_bins = int(valid_length[i].asscalar())-2
                    bin_counts_per_ctx[ctx_idx][num_bins, masked_positions[i]-1] += 1
                    bin_sums_per_ctx[ctx_idx][num_bins, masked_positions[i]-1] += out[i, 0, token_masked_ids[i]]
                    if token_masked_ids[i].asscalar() == 1012:
                        import pdb; pdb.set_trace()
                ### </DIFFERENT>

            # Progress
            sent_count += batch_size
            if (batch_id+1) % batch_log_interval == 0:
                logging.info("{} sents of {}, batch {} of {}".format(sent_count, len(dataset), batch_id+1, len(batch_sampler)))

        # Accumulate the counts
        for ctx_idx in range(len(self._ctxs)):
            bin_counts += bin_counts_per_ctx[ctx_idx].asnumpy()
            bin_sums += bin_sums_per_ctx[ctx_idx].asnumpy()

        return bin_counts, bin_sums


class RegressionFinetuner(BaseScorer):

    def __init__(self, *args, **kwargs):
        self._wwm = kwargs.pop('wwm') if 'wwm' in kwargs else False
        super().__init__(*args, **kwargs)
        # Turn Dataset into Dataloader
        self._batchify_fn = btf.Tuple(btf.Stack(dtype='int32'), btf.Pad(pad_val=np.iinfo(np.int32).max, dtype='int32'),
                              btf.Stack(dtype='float32'), btf.Stack(dtype='float32'))
        self._trainer = mx.gluon.Trainer(self._model.collect_params(), 'adam',
                           {'learning_rate': 1e-5, 'epsilon': 1e-9}, update_on_kvstore=False)
        self._loss = mx.gluon.loss.L2Loss()
        self._loss.hybridize(static_alloc=True)
        self._params = [p for p in self._model.collect_params().values() if p.grad_req != 'null']

        self._max_length = 384


    def corpus_to_dataset(self, corpus: ScoredCorpus) -> SimpleDataset:

        sents_expanded = []

        for sent_idx, sent_dict in enumerate(corpus.values()):
            sent = self._apply_tokenizer_opts(sent_dict['text'])
            tokens_original = [self._vocab.cls_token] + self._tokenizer(sent) + [self._vocab.eos_token]
            ids_original = np.array(self._tokenizer.convert_tokens_to_ids(tokens_original))

            # Enforce max length
            if len(ids_original) > self._max_length:
                logging.error("Line #{} is too long; will output score of 0 and omit in token counts (but not yet in word counts!)".format(sent_idx+1))
            else:
                sents_expanded += [(sent_idx, ids_original, len(ids_original), sent_dict['score'])]

        return SimpleDataset(sents_expanded)


    def _batch_ops(self, batch, batch_sent_idxs_per_ctx, batch_scores_per_ctx, temp) -> int:

        batch_size = 0

        losses = []

        with mx.autograd.record():

            for ctx_idx, (sent_idxs, token_ids, valid_length, scores) in enumerate(batch):

                ctx = self._ctxs[ctx_idx]
                batch_size += sent_idxs.shape[0]
                token_ids = token_ids.as_in_context(ctx)
                valid_length = valid_length.as_in_context(ctx)
                scores = scores.as_in_context(ctx)
                segment_ids = mx.nd.zeros(shape=token_ids.shape, ctx=ctx)
                out = self._model(token_ids, segment_ids, valid_length)
                loss = self._loss(out, scores).sum()

                losses.append(loss)
            
            for loss in losses:
                loss.backward()

        # Synchronous
        batch_loss = sum([loss.as_in_context(mx.cpu()) for loss in losses])
        losses = []

        # Gradient clipping
        self._trainer.allreduce_grads()
        nlp.utils.clip_grad_global_norm(self._params, 1)
        # TODO:  What is correct # of steps?
        # TODO: Stale grad?
        mx.nd.waitall()
        self._trainer.update(batch_size, ignore_stale_grad=True)

        return batch_size, batch_loss


    def tune(self, scored_corpus: ScoredCorpus, temp: float = 1.0, split_size: int = 2000, ratio: float = 0, num_workers: int = 10, output_dir: Optional[Path] = None) -> List[float]:

        ctx_cpu = mx.Context('cpu')

        # Get MXNet data objects
        dataset, batch_sampler, dataloader = self._corpus_to_data(scored_corpus, split_size, ratio, num_workers, shuffle=True)

        # Get number of tokens
        true_tok_lens = self._true_tok_lens(dataset)

        batch_log_interval = 500
        num_epochs = 10

        # For now just predicts the first non-cls token
        for epoch_id in range(num_epochs):

            sent_count = 0
            epoch_loss = 0.0

            for batch_id, batch in enumerate(dataloader):

                batch = self._split_batch(batch)

                batch_size, batch_loss = self._batch_ops(batch, None, None, temp)

                # Progress
                sent_count += batch_size
                epoch_loss += batch_loss
                if (batch_id+1) % batch_log_interval == 0:
                    logging.info("{} sents of {}, batch {} of {}".format(sent_count, len(dataset), batch_id+1, len(batch_sampler)))
                    logging.warning("Average squared loss = {}".format(batch_loss.asscalar() / batch_size))

            logging.warning("Finished epoch {}".format(epoch_id))
            logging.warning("Epoch loss: {}".format(epoch_loss / sent_count))
            self._model.save_parameters(str(output_dir / 'epoch-{}.params'.format(epoch_id+1)))

        return true_tok_lens


class RegressionScorer(BaseScorer):

    def __init__(self, *args, **kwargs):
        self._wwm = kwargs.pop('wwm') if 'wwm' in kwargs else False
        super().__init__(*args, **kwargs)
        # Turn Dataset into Dataloader
        self._batchify_fn = btf.Tuple(btf.Stack(dtype='int32'), btf.Pad(pad_val=np.iinfo(np.int32).max, dtype='int32'),
                              btf.Stack(dtype='float32'))
        # self._max_length = 256


    # Almost as RegressionFinetuner, except now it's just Corpus
    def corpus_to_dataset(self, corpus: Corpus) -> SimpleDataset:

        sents_expanded = []

        for sent_idx, sent in enumerate(corpus.values()):
            sent = self._apply_tokenizer_opts(sent)
            tokens_original = [self._vocab.cls_token] + self._tokenizer(sent) + [self._vocab.eos_token]
            ids_original = np.array(self._tokenizer.convert_tokens_to_ids(tokens_original))

            # Enforce max length
            if len(ids_original) > self._max_length:
                logging.error("Line #{} is too long; will output score of 0 and omit in token counts (but not yet in word counts!)".format(sent_idx+1))
            else:
                sents_expanded += [(sent_idx, ids_original, len(ids_original))]

        return SimpleDataset(sents_expanded)


    def _batch_ops(self, batch, batch_sent_idxs_per_ctx, batch_scores_per_ctx, temp) -> int:

        batch_size = 0

        for ctx_idx, (sent_idxs, token_ids, valid_length) in enumerate(batch):

            ctx = self._ctxs[ctx_idx]
            batch_size += sent_idxs.shape[0]
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = mx.nd.zeros(shape=token_ids.shape, ctx=ctx)
            # out is (batch size, )
            out = self._model(token_ids, segment_ids, valid_length)

            batch_sent_idxs_per_ctx[ctx_idx].append(sent_idxs)
            batch_scores_per_ctx[ctx_idx].append(out.reshape(-1))

        return batch_size
