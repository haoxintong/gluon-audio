# MIT License
# Copyright (c) 2019 haoxintong
""""""
import unittest

import mxnet as mx
import numpy as np
import scipy.fftpack
import librosa as rosa

from gluonar import nn


class UnittestBase(unittest.TestCase):
    def test_librosa_consistency(self, active=False):
        """Base class for Block Tests, this should be always success."""

    def test_hybridize_forward(self):
        self.test_librosa_consistency(active=True)


class TestSTFT(UnittestBase):
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

    def test_librosa_consistency(self, active=False):
        self.stft.hybridize(active)
        x = mx.nd.random_uniform(-1, 1, shape=(1, self.audio_length), ctx=mx.gpu(0))
        gluon_spec = self.stft(x).asnumpy()[0][0].transpose((1, 0))

        spec = rosa.stft(x.asnumpy()[0], **self.stft_params)
        spec = np.abs(spec)[:int(self.stft_params["n_fft"] / 2), ::]

        mx.test_utils.assert_almost_equal(gluon_spec, spec, atol=1e-5)


class TestDCT1D(UnittestBase):
    """
    Here we can only get a desired precision of 1e-4.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.signal_length = 512
        cls.dct = nn.DCT1D(N=cls.signal_length)
        cls.dct.initialize(ctx=mx.gpu(0))

    def test_librosa_consistency(self, active=False):
        self.dct.hybridize(active)
        dim = np.random.randint(1, 5)
        shape = np.random.randint(1, 10, size=(dim,))
        x = mx.nd.random_uniform(-1, 1, shape=(*shape, self.signal_length), ctx=mx.gpu(0))

        gluon_ret = self.dct(x).asnumpy()
        scipy_ret = scipy.fftpack.dct(x.asnumpy())

        mx.test_utils.assert_almost_equal(gluon_ret, scipy_ret, atol=1e-4)


class TestMelSpec(UnittestBase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.signal_length = 48000
        cls.mel_params = {
            "sr": 16000,
            "n_fft": 2048,
            "hop_length": 512,
            "power": 2.0,
            "n_mels": 128,
            "fmin": 0.0,
            "fmax": None,
            "htk": False,
            "norm": 1
        }
        cls.melspec = nn.MelSpectrogram(cls.signal_length, **cls.mel_params)
        cls.melspec.initialize(ctx=mx.gpu(0))

    def test_librosa_consistency(self, active=False):
        self.melspec.hybridize(active)
        x = mx.nd.random_uniform(-1, 1, shape=(1, self.signal_length), ctx=mx.gpu(0))
        gluon_ret = self.melspec(x).asnumpy()[0][0]

        rosa_ret = rosa.feature.melspectrogram(x.asnumpy()[0], **self.mel_params)
        rosa_ret = np.abs(rosa_ret)[:int(self.mel_params["n_fft"] / 2), ::]

        mx.test_utils.assert_almost_equal(gluon_ret, rosa_ret, atol=1e-5)


class TestPowerToDB(UnittestBase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.batch_size = 8
        cls.power2db = nn.PowerToDB()
        cls.power2db.initialize()

    def test_librosa_consistency(self, active=False):
        self.power2db.hybridize(active)
        x = np.random.uniform(-1, 1, (self.batch_size, 16000))

        specs = []
        rosa_ret = []
        for i in range(self.batch_size):
            spec = rosa.feature.melspectrogram(x[i])
            specs.append(spec)
            rosa_ret.append(rosa.power_to_db(spec))

        gluon_ret = self.power2db(mx.nd.array(specs)).asnumpy()

        mx.test_utils.assert_almost_equal(gluon_ret, np.array(rosa_ret), atol=1e-20)


class TestMFCC(UnittestBase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.signal_length = 48000
        cls.mfcc_params = {
            "sr": 16000,
            "n_mfcc": 20,
            "dct_type": 2,
            "norm": None,
            "n_fft": 2048,
            "hop_length": 512,
            "power": 2.0,
            "n_mels": 128,
            "fmin": 0.0,
            "fmax": None
        }
        cls.mfcc = nn.MFCC(cls.signal_length, **cls.mfcc_params)
        cls.mfcc.initialize(ctx=mx.gpu(0))

    def test_librosa_consistency(self, active=False):
        self.mfcc.hybridize(active)
        x = mx.nd.random_uniform(-1, 1, shape=(1, self.signal_length), ctx=mx.gpu(0))
        rosa_ret = rosa.feature.mfcc(x.asnumpy()[0], **self.mfcc_params)
        gluon_ret = self.mfcc(x).asnumpy()[0][0]

        mx.test_utils.assert_almost_equal(gluon_ret, rosa_ret, atol=1e-5)
