import collections
import functools
import mxnet as mx
import numpy as np
import scipy
import pytest
from mxnet.gluon import nn, HybridBlock
from numpy.testing import assert_allclose
from gluonnlp.sequence_sampler import BeamSearchScorer, BeamSearchSampler
mx.npx.set_np()


@pytest.mark.parametrize('length', [False, True])
@pytest.mark.parametrize('alpha', [0.0, 1.0])
@pytest.mark.parametrize('K', [1.0, 5.0])
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('vocab_size', [2, 5])
@pytest.mark.parametrize('from_logits', [False, True])
@pytest.mark.parametrize('hybridize', [False, True])
def test_beam_search_score(length, alpha, K, batch_size, vocab_size, from_logits, hybridize):
    scorer = BeamSearchScorer(alpha=alpha, K=K, from_logits=from_logits)
    if hybridize:
        scorer.hybridize()
    sum_log_probs = mx.np.zeros((batch_size,))
    scores = mx.np.zeros((batch_size,))
    for step in range(1, length + 1):
        if not from_logits:
            log_probs = np.random.normal(0, 1, (batch_size, vocab_size))
            log_probs = np.log((scipy.special.softmax(log_probs, axis=-1)))
        else:
            log_probs = np.random.uniform(-10, 0, (batch_size, vocab_size))
        log_probs = mx.np.array(log_probs, dtype=np.float32)
        sum_log_probs += log_probs[:, 0]
        scores = scorer(log_probs, scores, mx.np.array(step))[:, 0]
    lp = (K + length) ** alpha / (K + 1) ** alpha
    assert_allclose(scores.asnumpy(), sum_log_probs.asnumpy() / lp, 1E-5, 1E-5)


# TODO(sxjscience) Test for the state_batch_axis
@pytest.mark.parametrize('early_return', [False, True])
@pytest.mark.parametrize('eos_id', [0, None])
def test_beam_search(early_return, eos_id):
    class SimpleStepDecoder(HybridBlock):
        def __init__(self, vocab_size=5, hidden_units=4):
            super().__init__()
            self.x2h_map = nn.Embedding(input_dim=vocab_size, output_dim=hidden_units)
            self.h2h_map = nn.Dense(units=hidden_units, flatten=False)
            self.vocab_map = nn.Dense(units=vocab_size, flatten=False)

        @property
        def state_batch_axis(self):
            return 0

        @property
        def data_batch_axis(self):
            return 0

        def hybrid_forward(self, F, data, state):
            """

            Parameters
            ----------
            F
            data :
                (batch_size,)
            states :
                (batch_size, C)

            Returns
            -------
            out :
                (batch_size, vocab_size)
            new_state :
                (batch_size, C)
            """
            new_state = self.h2h_map(state)
            out = self.vocab_map(self.x2h_map(data) + new_state)
            return out, new_state

    vocab_size = 3
    batch_size = 2
    hidden_units = 3
    beam_size = 4
    step_decoder = SimpleStepDecoder(vocab_size, hidden_units)
    step_decoder.initialize()
    sampler = BeamSearchSampler(beam_size=4, decoder=step_decoder, eos_id=eos_id, vocab_size=vocab_size,
                                max_length_b=100, early_return=early_return)
    states = mx.np.random.normal(0, 1, (batch_size, hidden_units))
    inputs = mx.np.random.randint(0, vocab_size, (batch_size,))
    samples, scores, valid_length = sampler(inputs, states)
    samples = samples.asnumpy()
    valid_length = valid_length.asnumpy()
    for i in range(batch_size):
        for j in range(beam_size):
            vl = valid_length[i, j]
            if eos_id is not None:
                assert samples[i, j, vl - 1] == eos_id
            if vl < samples.shape[2]:
                assert (samples[i, j, vl:] == -1).all()
            assert (samples[i, :, 0] == inputs[i].asnumpy()).all()


# TODO(sxjscience) Test for the state_batch_axis
@pytest.mark.parametrize('early_return', [False, True])
@pytest.mark.parametrize('eos_id', [0, None])
def test_beam_search_stochastic(early_return, eos_id):
    class SimpleStepDecoder(HybridBlock):
        def __init__(self, vocab_size=5, hidden_units=4):
            super().__init__()
            self.x2h_map = nn.Embedding(input_dim=vocab_size, output_dim=hidden_units)
            self.h2h_map = nn.Dense(units=hidden_units, flatten=False)
            self.vocab_map = nn.Dense(units=vocab_size, flatten=False)

        @property
        def state_batch_axis(self):
            return 0

        @property
        def data_batch_axis(self):
            return 0

        def hybrid_forward(self, F, data, state):
            """

            Parameters
            ----------
            F
            data :
                (batch_size,)
            states :
                (batch_size, C)

            Returns
            -------
            out :
                (batch_size, vocab_size)
            new_state :
                (batch_size, C)
            """
            new_state = self.h2h_map(state)
            out = self.vocab_map(self.x2h_map(data) + new_state)
            return out, new_state

    vocab_size = 3
    batch_size = 2
    hidden_units = 3
    beam_size = 4
    step_decoder = SimpleStepDecoder(vocab_size, hidden_units)
    step_decoder.initialize()
    sampler = BeamSearchSampler(beam_size=4, decoder=step_decoder, eos_id=eos_id, vocab_size=vocab_size,
                                stochastic=True, max_length_b=100, early_return=early_return)
    states = mx.np.random.normal(0, 1, (batch_size, hidden_units))
    inputs = mx.np.random.randint(0, vocab_size, (batch_size,))
    samples, scores, valid_length = sampler(inputs, states)
    samples = samples.asnumpy()
    valid_length = valid_length.asnumpy()
    for i in range(batch_size):
        for j in range(beam_size):
            vl = valid_length[i, j]
            if eos_id is not None:
                assert samples[i, j, vl-1] == eos_id
            if vl < samples.shape[2]:
                assert (samples[i, j, vl:] == -1).all()
            assert (samples[i, :, 0] == inputs[i].asnumpy()).all()

    # test for repeativeness
    has_different_sample = False
    for _ in range(10):
        new_samples, scores, valid_length = sampler(inputs, states)
        if not np.array_equal(new_samples.asnumpy(), samples):
            has_different_sample = True
            break
    assert has_different_sample

@pytest.mark.parametrize('early_return', [False, True])
@pytest.mark.parametrize('sampling_paras', [(-1.0, -1), (0.05, -1), (-1.0, 1), (-1.0, 3)])
@pytest.mark.parametrize('eos_id', [0, None])
def test_multinomial_sampling(early_return, sampling_paras, eos_id):
    class SimpleStepDecoder(HybridBlock):
        def __init__(self, vocab_size=5, hidden_units=4):
            super().__init__()
            self.x2h_map = nn.Embedding(input_dim=vocab_size, output_dim=hidden_units)
            self.h2h_map = nn.Dense(units=hidden_units, flatten=False)
            self.vocab_map = nn.Dense(units=vocab_size, flatten=False)

        @property
        def state_batch_axis(self):
            return 0

        @property
        def data_batch_axis(self):
            return 0

        def hybrid_forward(self, F, data, state):
            new_state = self.h2h_map(state)
            out = self.vocab_map(self.x2h_map(data) + new_state)
            return out, new_state

    vocab_size = 5
    batch_size = 2
    hidden_units = 3
    beam_size = 4
    step_decoder = SimpleStepDecoder(vocab_size, hidden_units)
    step_decoder.initialize()
    sampling_topp, sampling_topk = sampling_paras
    sampler = BeamSearchSampler(beam_size=4, decoder=step_decoder, eos_id=eos_id, vocab_size=vocab_size,
                                stochastic=False,
                                sampling=True, sampling_topp=sampling_topp, sampling_topk=sampling_topk,
                                max_length_b=100, early_return=early_return)
    states = mx.np.random.normal(0, 1, (batch_size, hidden_units))
    inputs = mx.np.random.randint(0, vocab_size, (batch_size,))
    samples, scores, valid_length = sampler(inputs, states)
    samples = samples.asnumpy()
    valid_length = valid_length.asnumpy()
    for i in range(batch_size):
        for j in range(beam_size):
            vl = valid_length[i, j]
            if eos_id is not None:
                assert samples[i, j, vl - 1] == eos_id
            if vl < samples.shape[2]:
                assert (samples[i, j, vl:] == -1).all()
            assert (samples[i, :, 0] == inputs[i].asnumpy()).all()
