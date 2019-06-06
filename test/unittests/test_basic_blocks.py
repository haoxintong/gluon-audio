# MIT License
# Copyright (c) 2019 haoxintong
""""""
import unittest

import mxnet as mx
import numpy as np
import scipy.fftpack
import librosa as rosa

from gluonar import nn


class TestSTFT(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stft_params = {
            "n_fft": 2048,
            "hop_length": 160,
            "win_length": 400,
            "window": "hann",
            "center": True
        }
        cls.audio_length = 48000
        cls.stft = nn.STFTBlock(cls.audio_length, **cls.stft_params)
        cls.stft.initialize(ctx=mx.gpu(0))

    def test_gpu(self):
        data = mx.nd.ones((2, 3), mx.gpu(0))
        self.assertEqual(data.context, mx.gpu(0))
        self.assertEqual(data.shape, (2, 3))

    def test_common(self):
        batch_size = np.random.randint(1, 100)
        if self.stft_params["center"]:
            num_frames = 1 + int(self.audio_length / self.stft_params["hop_length"])
        else:
            num_frames = 1 + int((self.audio_length - self.stft_params["n_fft"]) / self.stft_params["hop_length"])

        x = mx.nd.random_uniform(-1, 1, shape=(batch_size, self.audio_length), ctx=mx.gpu(0))
        spec = self.stft(x)
        self.assertTrue(isinstance(spec, mx.nd.NDArray))
        self.assertEqual(spec.shape, (batch_size, 1, num_frames, int(self.stft_params["n_fft"] / 2)))

    def test_librosa_consistency(self):
        x = mx.nd.random_uniform(-1, 1, shape=(1, self.audio_length), ctx=mx.gpu(0))
        gluon_spec = self.stft(x).asnumpy()[0][0].transpose((1, 0))

        spec = rosa.stft(x.asnumpy()[0], **self.stft_params)
        spec = np.abs(spec)[:int(self.stft_params["n_fft"] / 2), ::]

        mx.test_utils.assert_almost_equal(gluon_spec, spec, rtol=1e-5, atol=1e-5)


class TestDCT1D(unittest.TestCase):
    """
    Here we can only get a desired precision of 1e-4.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.signal_length = 512
        cls.dct = nn.DCT1D(N=cls.signal_length)
        cls.dct.initialize(ctx=mx.gpu(0))

    def test_librosa_consistency(self):
        dim = np.random.randint(1, 5)
        shape = np.random.randint(1, 10, size=(dim,))
        x = mx.nd.random_uniform(-1, 1, shape=(*shape, self.signal_length), ctx=mx.gpu(0))

        gluon_ret = self.dct(x).asnumpy()
        scipy_ret = scipy.fftpack.dct(x.asnumpy())

        np.testing.assert_almost_equal(gluon_ret, scipy_ret, decimal=4)