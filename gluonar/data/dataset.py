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
"""Audio Recognition Dataset"""
import os
import warnings
import numpy as np
import librosa as rosa

from mxnet import nd
from mxnet.gluon.data import Dataset

from av import container

__all__ = ["VoxAudioFolderDataset", "VoxAudioValFolderDataset"]

format_dtypes = {
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

format_scale = {
    'flt': 1,
    'fltp': 1,
    's16': 2 ** 15,
    's16p': 2 ** 15,
    's32': 2 ** 31,
    's32p': 2 ** 31
}


def _load(path):
    if path.lower().endswith(".wav"):
        audio = rosa.load(path, sr=16000)[0]
    else:
        fin = container.open(path)
        audio_frames = [frame for frame in fin.decode()]
        audios = list(map(lambda x: np.frombuffer(x.planes[0], format_dtypes[x.format.name],
                                                  x.samples), audio_frames))
        audio = np.concatenate(audios, axis=0)
    return nd.array(audio)


class VoxAudioFolderDataset(Dataset):
    """Load an audio file .

    Parameters
    ----------
    root : string
        path to vox root.
    sr   : int, default is 16k.
        target sampling rate
    min_length: int, default is 3.
        min length audio required.

    """

    def __init__(self, root, sr=16000, min_length=3, transform=None):
        self._root = os.path.expanduser(root)
        self._sr = sr
        self._transform = transform
        self._min_length = min_length
        self._exts = ['.wav', '.m4a']
        self._list_audios(self._root)
        self.num_classes = len(self.synsets)

    def _list_audios(self, root):
        self.synsets = []
        self.items = []

        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
                continue

            label = len(self.synsets)
            self.synsets.append(folder)
            for subfolder in sorted(os.listdir(path)):
                subpath = os.path.join(path, subfolder)

                if not os.path.isdir(subpath):
                    warnings.warn('Ignoring %s, which is not a directory.' % subpath, stacklevel=3)
                    continue
                for filename in sorted(os.listdir(subpath)):
                    filename = os.path.join(subpath, filename)
                    ext = os.path.splitext(filename)[1]
                    if ext.lower() not in self._exts:
                        warnings.warn('Ignoring %s of type %s. Only support %s' % (
                            filename, ext, ', '.join(self._exts)))
                        continue
                    self.items.append((filename, label))

    def __getitem__(self, idx):
        while True:
            audio = _load(self.items[idx][0])
            if audio.shape[0] < self._sr * self._min_length:
                idx = np.random.randint(low=0, high=len(self))
                continue
            label = self.items[idx][1]
            if self._transform is not None:
                return self._transform(audio, label)
            return audio, label

    def __len__(self):
        return len(self.items)


class VoxAudioValFolderDataset(Dataset):
    """

    Parameters
    ----------
    lst_path : str. Path of Val Audio list.
    root : str. Path to face folder. Default is '$(HOME)/.mxnet/datasets/sound'
    transform : callable, default None
        A function that takes data and transforms them:
    ::
        transform = lambda data: data.astype(np.float32)/255

    """

    def __init__(self, lst_path, root=os.path.expanduser('~/.mxnet/datasets/sound'),
                 sr=16000, transform=None):
        super().__init__()
        self._transform = transform
        self._sr = sr
        self._items, self._issame_list = [], []
        with open(lst_path, 'r') as f:
            for line in f.readlines():
                tmp = line.strip().split(" ")
                self._issame_list.append(int(tmp[0]))
                self._items.append((os.path.join(root, tmp[1]),
                                    os.path.join(root, tmp[2])))

    def __getitem__(self, idx):
        audio0 = _load(self._items[idx][0])
        audio1 = _load(self._items[idx][1])
        issame = self._issame_list[idx]

        if self._transform is not None:
            audio0 = self._transform(audio0)
            audio1 = self._transform(audio1)

        return (audio0, audio1), issame

    def __len__(self):
        return len(self._items)


class TIMITDataset(Dataset):
    def __init__(self, root, is_train=True, sr=16000, min_length=3, transform=None):
        self._sr = sr
        self._transform = transform
        self._min_length = min_length
        self._exts = ['.wav', '.m4a']
        _root = os.path.expanduser(root)
        self._root = os.path.join(_root, 'TRAIN' if is_train else 'TEST')
        self._list_audios(self._root)
        self.num_classes = len(self.synsets)

    def _list_audios(self, root):
        self.synsets = []
        self.items = []

        for folder_dr in sorted(os.listdir(root)):
            path = os.path.join(root, folder_dr)
            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
                continue
            for folder_idt in sorted(os.listdir(path)):
                audio_root = os.path.join(path, folder_idt)
                if not os.path.isdir(audio_root):
                    warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
                    continue
                label = len(self.synsets)
                self.synsets.append(folder_idt)
                for fn in sorted(os.listdir(audio_root)) :
                    sound_fp = os.path.join(audio_root, fn)
                    ext = os.path.splitext(sound_fp)[-1]
                    if ext.lower() not in self._exts:
                        continue
                    self.items.append((sound_fp, label))

    def __getitem__(self, idx):
        while True:
            audio = _load(self.items[idx][0])
            if audio.shape[0] < self._sr * self._min_length:
                idx = np.random.randint(low=0, high=len(self))
                continue
            label = self.items[idx][1]
            if self._transform is not None:
                return self._transform(audio, label)
            return audio, label

    def __len__(self):
        return len(self.items)
