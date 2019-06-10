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
from librosa import filters
from mxnet import init, nd
from mxnet.gluon import nn

__all__ = ["SincConv1D", "ZScoreNormBlock", "STFTBlock", "DCT1D", "MelSpectrogram", "MFCC",
           "PowerToDB"]


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
    audio_length: int.
        target audio length.

    Inputs:
        - **x**: the input audio signal, with shape (batch_size, audio_length).

    Outputs:
        - **specs**: specs tensor with shape (batch_size, 1, num_frames, win_lengths/2).

    Notes
    -----
    The output shape is calculated by (1+(len(y)-n_fft)/hop_length, win_lengths/2),
    and different from librosa the output should be transposed before visualization.
    """

    def __init__(self, audio_length, n_fft=2048, hop_length=None,
                 win_length=None, window="hann", center=True, power=1.0, **kwargs):
        super().__init__(**kwargs)

        if win_length is None:
            win_length = n_fft

        if hop_length is None:
            hop_length = int(win_length // 4)
        if center:
            num_frames = 1 + int(audio_length / hop_length)
        else:
            num_frames = 1 + int((audio_length - n_fft) / hop_length)
        win = signal.get_window(window, win_length)
        win = np.expand_dims(self.pad_center(win, n_fft), axis=0)
        indices = np.tile(np.arange(0, n_fft), (num_frames, 1)) + np.tile(np.arange(0, num_frames * hop_length,
                                                                                    hop_length), (n_fft, 1)).T
        self._audio_length = int(audio_length)
        self._n_fft = n_fft
        self._win_length = win_length
        self._num_frames = num_frames
        self._power = power
        self._center = center
        with self.name_scope():
            self.window = self.params.get_constant("window", win)
            self.indices = self.params.get_constant("indices", indices)

    def hybrid_forward(self, F, x, window, indices, *args, **kwargs):
        if self._center:
            frames = self.pad(F, x)
        else:
            frames = x

        frames = F.take(frames, indices, axis=1)
        frames = F.broadcast_mul(frames, window)
        frames = F.reshape(frames, shape=(-1, self._n_fft))

        specs = F.contrib.fft(frames)
        specs = F.reshape(specs, shape=(-1, self._num_frames, self._n_fft, 2))
        specs = F.slice(specs, begin=(None, None, 0, None), end=(None, None, int(self._n_fft / 2), None))

        specs = F.power(self.abs_complex(F, specs, axis=3), self._power)
        return F.expand_dims(specs, axis=1)

    def pad(self, F, x):
        """As mxnet nd.pad only support 4D or 5D arrays, it should be reshaped first."""
        x_4d = F.reshape(x, shape=(1, 1, -1, self._audio_length))
        x_4d = F.pad(x_4d, mode="reflect", pad_width=(0, 0, 0, 0, 0, 0, int(self._n_fft // 2), int(self._n_fft // 2)))
        return F.reshape(x_4d, shape=(-1, self._audio_length + int(self._n_fft // 2 * 2)))

    @staticmethod
    def abs_complex(F, x, axis):
        """Calculate the absolute value element-wise along given axis."""
        return F.sqrt(F.sum(F.square(x), axis=axis))

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
    """Zero Score Normalization Block"""

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
        norm_x = F.broadcast_div(F.broadcast_sub(x, F.expand_dims(mean, -1)), F.expand_dims(std + 1e-7, -1))
        return norm_x


class DCT1D(nn.HybridBlock):
    """
    Compute the Discrete Cosine Transform of input data. This block is implemented
    as scipy.

    `DCT1D`'s behavior is compute dct along last axis for any dimensions larger than 2.

    Parameters
    ----------
    mode : {1, 2, 3}, optional
        Type of the DCT (see Notes). Default type is 2.
    N : int
        Length of the transform. The required value is ``N = x.shape[axis]``.
    axis : int, optional
        Axis along which the dct is computed; the default is over the
        last axis (i.e., ``axis=-1``).
    norm : {None, 'ortho'}, optional
        Normalization mode (see Notes). Default is None.

    Notes
    -----
    **Type II**

    There are several definitions of the DCT-II; use scipy definition following
    (for ``norm=None``)::

                N-1
      y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
                n=0

    If ``norm='ortho'``, ``y[k]`` is multiplied by a scaling factor `f`::

      f = sqrt(1/(4*N)) if k = 0,
      f = sqrt(1/(2*N)) otherwise.

    Which makes the corresponding matrix of coefficients orthonormal
    (``OO' = Id``).

    TODO: implement the norm coef option.
    """

    def __init__(self, mode=2, N=None, norm=None, **kwargs):
        super().__init__(**kwargs)

        if mode != 2:
            raise NotImplementedError("DCT types other than II not yet supported")

        if N is None:
            raise ValueError("Signal length is required by DCT1D, fill N with it.")

        coef = np.multiply(np.array([(2 * n + 1) / (2 * N) for n in range(N)]).reshape([N, 1]),
                           np.arange(0, N, 1).reshape([1, N]))
        coef = np.cos(coef * np.pi)

        self.coef = self.params.get_constant("coef", coef)

    def hybrid_forward(self, F, x, coef, *args, **kwargs):
        return 2 * F.dot(x, coef)


class MelSpectrogram(nn.HybridBlock):
    """Compute a mel-scaled spectrogram.

    Parameters
    ----------
    audio_length: int.
        target audio length.
    sr : number > 0 [scalar]
        sampling rate of `audio`

    n_fft : int > 0 [scalar]
        length of the FFT window

    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `librosa.core.stft`

    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram.
        e.g., 1 for energy, 2 for power, etc.

    others : additional arguments
      Mel filter bank parameters.
      See `librosa.filters.mel` for details.

    """

    def __init__(self, audio_length, sr=16000, n_fft=2048, hop_length=512, power=2.0,
                 n_mels=128, fmin=0.0, fmax=None, htk=False, norm=1, **kwargs):
        super().__init__(**kwargs)
        self.stft = STFTBlock(audio_length, n_fft, hop_length, power=power)
        mel_basis = filters.mel(sr, n_fft, n_mels, fmin, fmax, htk, norm)[::, :1024].transpose((1, 0))
        self.mel_basis = self.params.get_constant("mel_basis", mel_basis)

    def hybrid_forward(self, F, x, mel_basis, *args, **kwargs):
        spec = self.stft(x)
        return F.transpose(F.dot(spec, mel_basis), axes=(0, 1, 3, 2))


class PowerToDB(nn.HybridBlock):
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units.
    This is modified from librosa.power_to_db, and make it be able to process
    batch input.

    For input shape of (batch, channel, w, h) or (batch, w, h), this block will
    compute power to db along last 2 axis.

    Parameters
    ----------

    ref : float.
        The amplitude `abs(S)` is scaled relative to `ref`:
        `10 * log10(S / ref)`.

    amin : float > 0 [scalar]
        minimum threshold for `abs(S)` and `ref`

    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(10 * log10(S)) - top_db``

    """

    def __init__(self, ref=1.0, amin=1e-10, top_db=80.0, **kwargs):
        super().__init__(**kwargs)
        self.ref = ref
        self.amin = amin
        if top_db is not None:
            if top_db < 0:
                raise ValueError('top_db must be non-negative')

        self.top_db = top_db
        self.db_multiplier = float(10.0 * np.log10(np.maximum(self.amin, self.ref)))

    def hybrid_forward(self, F, x, *args, **kwargs):
        log_spec = 10.0 * F.log10(F.maximum(self.amin, x)) - self.db_multiplier

        if self.top_db is not None:
            log_spec = F.broadcast_maximum(log_spec, F.max(log_spec, axis=(-2, -1), keepdims=True) - self.top_db)
        return log_spec


class MFCC(nn.HybridBlock):
    """
    TODO: add DOC, fix numerical precision problem.
    """

    def __init__(self, audio_length, sr=16000, n_mfcc=20, dct_type=2, norm='ortho',
                 n_fft=2048, hop_length=512, power=2.0, n_mels=128, fmin=0.0, fmax=None, **kwargs):
        super().__init__(**kwargs)

        self.mel_spec = MelSpectrogram(audio_length, sr, n_fft, hop_length, power, n_mels, fmin, fmax)
        self.power2db = PowerToDB()
        self.dct = DCT1D(mode=dct_type, N=n_mels, norm=norm)
        self._n_mfcc = n_mfcc

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.mel_spec(x)
        x = F.transpose(self.power2db(x), (0, 1, 3, 2))
        x = F.transpose(self.dct(x), (0, 1, 3, 2))
        return F.slice(x, begin=(None, None, 0, None), end=(None, None, self._n_mfcc, None))
