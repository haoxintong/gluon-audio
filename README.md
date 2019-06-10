# Gluon Audio Toolkit
Gluon Audio is a toolkit providing deep learning based audio recognition algorithm. 
The project is still under development, and only Chinese introduction will be provided.

## GluonAR Introduction:
GluonAR is based on MXnet-Gluon, if you are new to it, please check out [dmlc 60-minute crash course](http://gluon-crash-course.mxnet.io/).
  
虽然名字叫GluonAR, 但是目前以及可以预见的时间内只有Text-Independent Speaker Recognition的内容.

已经实现的feature:
- 使用ffmpeg的pythonic binding `av`和`librosa`做audio数据读取
- 模块支持`Hybridize()`. forward阶段不使用pysound, librosa, scipy, 效率更高, 提供快速训练和end-to-end部署的能力, 包括:
    - 基于`nd.contrib.fft`的短时傅里叶变换(`STFTBlock`)和z-score block, 相比使用numpy和scipy预处理后载入GPU训练效率提高12%.
    - `MelSpectrogram`, `DCT1D`, `MFCC`, `PowerToDB`
    - [1808.00158](https://arxiv.org/abs/1808.00158)中提出的`SincBlock`
- gluon风格的VOX数据集载入
- 类似人脸验证的Speaker Verification
- 使用频谱图训练声纹特征的例子, 在VOX1上的1:1验证acc: 0.941152+-0.004926

example:
```python
import numpy as np
import mxnet as mx
import librosa as rosa
from gluonar.utils.viz import view_spec
from gluonar.nn.basic_blocks import STFTBlock

data = rosa.load(r"resources/speaker_recognition/speaker0_0.m4a", sr=16000)[0][:35840]
nd_data = mx.nd.array([data], ctx=mx.gpu())

stft = STFTBlock(35840, hop_length=160, win_length=400)
stft.initialize(ctx=mx.gpu())

# stft block forward
ret = stft(nd_data).asnumpy()[0][0]
spec = np.transpose(ret, (1, 0)) ** 2
view_spec(spec)

# stft in librosa 
spec = rosa.stft(data, hop_length=160, win_length=400, window="hamming")
spec = np.abs(spec) ** 2
view_spec(spec)
```
输出:

|STFTBlock| STFT in librosa |
|:---:|:---:|
|<img src="resources/spectrogram.png"/>|<img src="resources/librosa_spectrogram.png"/>|

更多的例子请参考`examples/`.

## Requirements
mxnet-1.5.0+, gluonfr, av, librosa, ...

音频库的选择主要考虑数据读取速度, 训练过程中音频的解码相比图像解码会消耗更多时间, 实际测试librosa从磁盘加载一个aac编码的短音频
耗时是pyav的8倍左右. 

- librosa  
    `pip install librosa`
- ffmpeg  
    ```
    # 下载ffmpeg源码, 进入根目录
    ./configure --extra-cflags=-fPIC --enable-shared
    make -j
    sudo make install
    ```
- pyav, 需要先安装ffmpeg  
    `pip install av`
- gluonfr  
    `pip install git+https://github.com/THUFutureLab/gluon-face.git@master`
    
## Datasets
### TIMIT
The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus (TIMIT) Training and Test Data.
Before using this dataset please follow the instruction on [link](https://catalog.ldc.upenn.edu/LDC93S1).

A copy of this was uploaded to [Google Drive](https://goo.gl/l0sPwz) by @philipperemy [here](https://github.com/philipperemy/timit).

### VoxCeleb
VoxCeleb is an audio-visual dataset consisting of short clips of human speech, extracted from interview videos uploaded to YouTube.  

For more information, checkout this [page](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

## Pretrained Models
### Speaker Recognition
#### ResNet18 training with VoxCeleb
Download: [Baidu](https://pan.baidu.com/s/1Gkhi67oJSiSyAiYNTdPlTw), 
[Google Drive](https://drive.google.com/open?id=1oEvSQrnNwYL4pRyQ8t87hRP3m22wuePz)

I followed the ideas in paper **VoxCeleb2** [1806.05622](https://arxiv.org/abs/1806.05622)  to train this model, 
the differences between them:

|-|Res18 in this repo|Res34 in paper|
|:---|:---:|:---:|
|Trained on|VoxCeleb2|VoxCeleb2|
|Input spec size|224x224|512x300|
|Eval on|Random 9500+ pair samples from VoxCeleb1 train and test set |Original VoxCeleb1 test set|
|Metric|Accuracy:0.932656+-0.005187|EER: 0.0504|
|Framework|Mxnet Gluon|Matconvnet|
|ROC|<img src="resources/vox_res18_roc.png" width="350"/>|-|

## TODO
接下来会慢慢补全使用mxnet gluon训练说话人识别的工具链, 预计会花超长时间. 

## Docs
GluonAR documentation is not available now. 

## Authors
{ [haoxintong](https://github.com/haoxintong) }

## Discussion
Any suggestions, please open an issue.

## Contributes
The final goal of this project is providing an easy using deep learning based audio algorithm library like 
[pytorch-kaldi](https://github.com/mravanelli/pytorch-kaldi).

Contribution is welcomed.

## References
1. MXNet Documentation and Tutorials [https://zh.diveintodeeplearning.org/](https://zh.diveintodeeplearning.org/)


