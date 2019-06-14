# MIT License
# Copyright (c) 2019 haoxintong
""""""
import os
import unittest
from mxnet.gluon.data import DataLoader
from gluonar.data import *


class TransformAudio:
    DTYPES = ("float32", "float16")

    def __init__(self, audio_length, dtype="float32"):
        self.audio_length = int(audio_length)
        if dtype not in self.DTYPES:
            raise ValueError("Dtype other than float32/16 is not supported.")
        self.dtype = dtype

    def train(self, data, label):
        data = data.astype(self.dtype)
        data = random_crop(data, self.audio_length)
        return data, label

    def val(self, data):
        data = data.astype(self.dtype)
        data = center_crop(data, self.audio_length)
        return data


class TestVoxAudioValFolderDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.data_root = os.path.expanduser("~/data/vox")
        trans = TransformAudio(48000)
        self.dataset = VoxAudioValFolderDataset(os.path.join(self.data_root, "sampled_pairs.txt"),

                                                root=os.path.join(self.data_root, "train1"),
                                                transform=trans.val)
        self.data_loader = DataLoader(self.dataset, 8, num_workers=4)

    def test_get_audio(self):
        for i, data in enumerate(self.dataset):
            audio0, audio1 = data[0]
            self.assertEqual(audio0.shape[0], audio1.shape[0], "{}th audio pairs got different shape!".format(i))
            if i > 10:
                break

    def test_data_loader(self):
        for i, batch in enumerate(self.data_loader):
            data_pairs = batch[0]
            self.assertEqual(data_pairs[0].shape, data_pairs[1].shape, "Shape not equal in a batch.")
            if i > 10:
                break
