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
"""A demo for speaker recognition."""
from av import container
import numpy as np
import mxnet as mx
from mxnet import gluon, nd


def compute_euclidean_dist(feature1, feature2):
    return np.sqrt(np.sum(np.square(np.subtract(feature1, feature2))))


class SpeakerVerification:
    """A demo to verify if the two input audio files is from same speaker.

    Parameters
    ----------
    threshold : float.
        path to vox root.
    json_path : str.
        symbol path of model, which contains the graphdef of mxnet model
    param_path : str.
        the params path of the model network.
    ctx :
        device context of network.
    audio_length: int, default is 35840.
        the audio length of input, here default value is calculated by 2.24*16000,
        that the audio last 2.24s and sample rate 16K, 35840 sample points in total.
    """
    def __init__(self, threshold, json_path, param_path, ctx=mx.gpu(), audio_length=35840):
        self._format_dtypes = {
            'dbl': '<f8',
            'dblp': '<f8',
            'flt': '<f4',
            'fltp': '<f4',
            's16': '<i2',
            's16p': '<i2',
            's32': '<i4',
            's32p': '<i4',
            'u8': 'u1',
            'u8p': 'u1',
        }
        self._audio_length = audio_length
        self._threshold = threshold
        self.ctx = ctx
        self.net = gluon.nn.SymbolBlock.imports(json_path, ["data"], param_path, self.ctx)
        self.net.summary(mx.nd.ones((1, 35840), mx.gpu()))
        self.net.hybridize()
        _ = self.net(mx.nd.ones((1, 35840), mx.gpu())).asnumpy()

    def load_audio(self, path):
        """This require an audio input longer than 2.24s."""
        fin = container.open(path)
        audio_frames = [frame for frame in fin.decode()]
        audios = list(map(lambda x: np.frombuffer(x.planes[0], self._format_dtypes[x.format.name],
                                                  x.samples), audio_frames))
        audio = np.concatenate(audios, axis=0)
        return nd.array(np.expand_dims(audio[:self._audio_length], axis=0))

    def forward(self, path):
        audio = self.load_audio(path).astype("float32").as_in_context(self.ctx)
        feature = self.net(audio)
        return nd.L2Normalization(feature, mode="instance")[0].asnumpy()

    def __call__(self, path1, path2):
        f1, f2 = self.forward(path1), self.forward(path2)
        dist = compute_euclidean_dist(f1, f2)
        return True if dist < self._threshold else False, dist


if __name__ == '__main__':
    import os
    from os.path import join as opj

    root = opj(os.path.dirname(__file__), "../..")
    demo_data = {"speaker0_0": opj(root, "resources/speaker_recognition/speaker0_0.m4a"),
                 "speaker0_1": opj(root, "resources/speaker_recognition/speaker0_1.m4a"),
                 "speaker1_0": opj(root, "resources/speaker_recognition/speaker1_0.m4a"),
                 "speaker1_1": opj(root, "resources/speaker_recognition/speaker1_1.m4a")}

    demo = SpeakerVerification(1.0,
                               opj(root, "models/speaker_recognition/vox-res18-symbol.json"),
                               opj(root, "models/speaker_recognition/vox-res18-0000.params"))

    for k, p0 in demo_data.items():
        print(k, "\t", end=" ")
        for _, p1 in demo_data.items():
            _, d = demo(p0, p1)
            print("{:.4f}".format(d), end=" ")
        print("\n")
