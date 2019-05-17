# MIT License
# Copyright (c) 2019 haoxintong
"""Some functions about audio processing and transform."""
import random
import numpy as np
import librosa as rosa

from mxnet import nd

__all__ = ["random_crop", "center_crop", "norm_audio"]


def random_crop(x, crop_length):
    length = x.shape[0]
    x0 = random.randint(0, length - crop_length)
    return nd.slice(x, begin=(x0,), end=(x0 + crop_length,))


def center_crop(x, crop_length):
    length = x.shape[0]
    x0 = int((length - crop_length) / 2)
    return nd.slice(x, begin=(x0,), end=(x0 + crop_length,))


def norm_audio(x, mean=nd.array([6.151717314181507e-05]), std=nd.array([0.06391325364253263])):
    return (x - mean) / std


def get_spec(signal, n_fft=600, hop_length=160, win_length=400):
    spec = rosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hamming")
    return np.abs(spec) ** 2
