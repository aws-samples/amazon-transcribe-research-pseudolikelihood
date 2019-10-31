import pytest

import mxnet as mx
from mxnet.gluon.data import Dataset

from lpl.loaders import Corpus
from lpl.models import get_pretrained
from lpl.scorers import LPLScorer



def _get_scorer_and_corpus():
    ctxs = [mx.gpu()]
    model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-en-uncased')
    scorer = LPLScorer(model, vocab, tokenizer, ctxs, eos=True, wwm=False)
    corpus = Corpus.from_dict({'utt': {'ref': "I am Sam"}})
    return scorer, corpus


def test_lplscorer_corpus_to_dataset():
    scorer, corpus = _get_scorer_and_corpus()
    dataset = scorer.corpus_to_dataset(corpus)
    assert isinstance(dataset, Dataset)
    # Our three tokens, plus the EOS
    assert len(dataset) == 4


def test_lplscorer_score():
    scorer, corpus = _get_scorer_and_corpus()
    scores, _ = scorer.score(corpus)
    assert len(scores) == 1
    assert pytest.approx(scores[0]) == -13.3046875
