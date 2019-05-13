# MIT License
#
# Copyright (c) 2019 haoxintong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Basic Blocks used in GluonAR."""

import math
import numpy as np
from scipy import signal
from mxnet import init
from mxnet.gluon import nn
from gluonfr.nn import *

__all__ = ['NormDense', 'SELayer', 'FrBase', 'SincConv1D', 'ZScoreNormBlock', 'STFTBlock']


class SincConv1D(nn.HybridBlock):
    """Sinc Conv Block from
    `"Speaker Recognition from Raw Waveform with SincNet"
    <https://arxiv.org/abs/1808.00158>`_ paper.

    """
    def __init__(self, channels, kernel_size, sample_rate, dtype='float32', **kwargs):
        super().__init__(**kwargs)
        # self._pi = math.pi

        self._sr = sample_rate
        self._channels = channels
        self._kernel_size = kernel_size
        self._min_freq = 50.0
        self._min_band = 50.0

        self._kwargs = {
            'kernel': kernel_size, 'stride': 1, 'dilate': 1,
            'pad': 0, 'num_filter': channels, 'num_group': 1,
            'no_bias': True, 'layout': "NCW"}

        f1_init, f2_init = self._init_weight(channels, sample_rate)

        N = self._kernel_size
        n = np.linspace(0, N, N)
        window = np.expand_dims(0.54 - 0.46 * np.cos(2 * math.pi * n / N), axis=0)
        t_right = np.linspace(1, (N - 1) / 2, int((N - 1) / 2)) / self._sr
        t_right_2_pi = np.expand_dims(t_right, 0) * 2 * math.pi

        with self.name_scope():
            self.window = self.params.get_constant("window", window)
            self.t_right_2_pi = self.params.get_constant("t_right_2_pi", t_right_2_pi)
            self.f1 = self.params.get("f1", shape=(channels, 1),
                                      init=init.Constant(f1_init.tolist()), dtype=dtype,
                                      allow_deferred_init=True)
            self.f2 = self.params.get("f2", shape=(channels, 1),
                                      init=init.Constant(f2_init.tolist()), dtype=dtype,
                                      allow_deferred_init=True)

    def _init_weight(self, channels, sample_rate):
        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, channels)  # Equally spaced in Mel scale
        f_cos = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        b1, b2 = np.roll(f_cos, 1), np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (sample_rate / 2) - 100
        f1_init = np.expand_dims(b1 / self._sr, 1)
        f2_init = np.expand_dims((b2 - b1) / self._sr, 1)
        return f1_init, f2_init

    def _sinc(self, F, band, t_right_2_pi):
        y_right = F.sin(F.broadcast_mul(band, t_right_2_pi)) / F.broadcast_mul(band, t_right_2_pi)
        y_left = F.flip(y_right, 1)
        return F.concat(y_left, F.ones((self._channels, 1)), y_right, dim=1)

    def hybrid_forward(self, F, x, f1, f2, window, t_right_2_pi, *args, **kwargs):
        # 本质上在构造带通滤波器, 利用逆向傅里叶变换在时域进行分析, 两个可学习参数表示高低截断频率

        f1 = F.abs(f1) + self._min_freq / self._sr
        f2 = f1 + (F.abs(f2) + self._min_band / self._sr)

        lp1 = 2 * F.broadcast_mul(f1, self._sinc(F, f1 * self._sr, t_right_2_pi))
        lp2 = 2 * F.broadcast_mul(f2, self._sinc(F, f2 * self._sr, t_right_2_pi))
        band_pass = lp2 - lp1

        weight = F.broadcast_div(band_pass, F.max(band_pass, axis=1, keepdims=True))
        weight = F.broadcast_mul(weight, window)
        weight = F.expand_dims(weight, axis=1)

        out = F.Convolution(x, weight, name='fwd', **self._kwargs)
        return out


class STFTBlock(nn.HybridBlock):
    """Short-Time Fourier Transform Block.

    Parameters
    ----------
    sample_rate: int, default is 16k.
        target sampling rate.

    Inputs:
        - **x**: the input audio signal, with shape (batch_size, audio_length).

    Outputs:
        - **specs**: specs tensor with shape (batch_size, 1, num_frames, win_lengths).
    """

    def __init__(self, audio_duration, sample_rate=16000, n_fft=600,
                 hop_length=160, win_length=400, window="hamming", **kwargs):
        super().__init__(**kwargs)
        if win_length is None:
            win_length = n_fft
        num_frames = 3 + int((audio_duration * sample_rate - win_length) / hop_length)

        win = signal.get_window(window, win_length)
        win = np.expand_dims(self.pad_center(win, n_fft), axis=0)
        indices = np.tile(np.arange(0, n_fft), (num_frames, 1)) + np.tile(np.arange(0, num_frames * hop_length,
                                                                                    hop_length), (n_fft, 1)).T

        self._n_fft = n_fft
        self._win_length = win_length
        self._num_frames = num_frames

        with self.name_scope():
            self.window = self.params.get_constant("window", win)
            self.indices = self.params.get_constant("indices", indices)

    def hybrid_forward(self, F, x, window, indices, *args, **kwargs):
        frames = F.take(x, indices, axis=1, mode="clip")
        frames = F.broadcast_mul(frames, window)
        frames = F.reshape(frames, shape=(-1, self._n_fft))
        specs = F.contrib.fft(frames)
        specs = F.reshape(specs, shape=(-1, self._num_frames, self._n_fft, 2))
        specs = F.sum(F.square(specs), axis=3)
        specs = F.expand_dims(specs, axis=1)
        specs = F.slice(specs, begin=(None, None, None, 0), end=(None, None, None, int(self._n_fft / 2)))
        return specs

    @staticmethod
    def pad_center(data, size, axis=-1, **kwargs):
        """Wrapper for np.pad to automatically center an array prior to padding.
        This is analogous to `str.center()`

        Parameters
        ----------
        data : np.ndarray
            Vector to be padded and centered

        size : int >= len(data) [scalar]
            Length to pad `data`
        """

        kwargs.setdefault('mode', 'constant')
        n = data.shape[axis]
        lpad = int((size - n) // 2)
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (lpad, int(size - n - lpad))
        if lpad < 0:
            raise ValueError(('Target size ({:d}) must be '
                              'at least input size ({:d})').format(size, n))

        return np.pad(data, lengths, **kwargs)


class ZScoreNormBlock(nn.HybridBlock):
    """Zero Score Normalization Block

    """
    def __init__(self, in_channels, in_shapes, **kwargs):
        super().__init__(**kwargs)
        self._in_channels = in_channels
        assert len(in_shapes) == 2, "feature shape should a tuple of 2 elements."
        self._in_h = in_shapes[0]
        self._in_w = in_shapes[1]

    def hybrid_forward(self, F, x, *args, **kwargs):
        t = F.reshape(x, (-1, self._in_channels, self._in_h * self._in_w))
        mean = F.mean(t, axis=-1, keepdims=True)
        std = F.sqrt(F.mean(F.square(F.broadcast_sub(t, mean)), axis=-1, keepdims=True))
        norm_x = F.broadcast_div(F.broadcast_sub(x, F.expand_dims(mean, -1)), F.expand_dims(std+1e-7, -1))
        return norm_x
