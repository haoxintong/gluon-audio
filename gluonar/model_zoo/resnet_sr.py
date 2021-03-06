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
""""""
from mxnet.gluon import nn
from mxnet.gluon.model_zoo.vision import get_resnet
from gluonfr.nn.basic_blocks import FrBase

from ..nn.basic_blocks import STFTBlock, ZScoreNormBlock


class ResNetSR(FrBase):
    def __init__(self, classes, num_layers, audio_length, n_fft=2048, hop_length=None, win_length=None,
                 weight_norm=False, feature_norm=False, embedding_size=128, need_cls_layer=True, **kwargs):
        super().__init__(classes, embedding_size, weight_norm,
                         feature_norm, need_cls_layer, **kwargs)
        spec_shape = (1 + int(audio_length / hop_length), int(n_fft / 2))
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='feature_')
            with self.features.name_scope():
                self.features.add(STFTBlock(audio_length, n_fft, hop_length, win_length),
                                  ZScoreNormBlock(1, spec_shape),
                                  nn.Activation("sigmoid"))
                self.features.add(get_resnet(2, num_layers, classes=embedding_size))
                self.features.add(nn.Dense(embedding_size, use_bias=False),
                                  nn.BatchNorm(scale=False, center=False),
                                  nn.PReLU())
