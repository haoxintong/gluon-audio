# Gluon Audio Toolkit
Gluon Audio is a toolkit providing deep learning based audio recognition algorithm. 
The project is still under development, and only Chinese introduction will be provided.

## GluonAR Introduction:
GluonAR is based on MXnet-Gluon, if you are new to it, please check out [dmlc 60-minute crash course](http://gluon-crash-course.mxnet.io/).
  
虽然名字叫GluonAR, 但是目前以及可以预见的时间内只有Text-Independent Speaker Recognition的内容.

已经实现的feature(一些实现依赖mxnet版本, 目前仍不稳定):
- 使用ffmpeg的封装库`av`和`librosa`做audio数据读取
- 基于`nd.contrib.fft`的短时傅里叶变换(`STFTBlock`) 
- z-score block
- [1808.00158](https://arxiv.org/abs/1808.00158)中提出的`SincBlock`
- gluon风格的VOX数据集载入
- 类似人脸验证的Speaker Verification
- 使用频谱图训练声纹特征的例子, 在VOX1上的1:1验证acc: 0.941152+-0.004926

## Requirements
mxnet-1.5.0+, gluonfr, av, librosa, ...

- librosa  
    `pip install librosa`
- ffmpeg  
    ```
    # 先下载ffmpeg源码, 进入根目录
    ./configure --extra-cflags=-fPIC --enable-shared
    make -j
    sudo make install
    ```
- pyav, 需要先安装ffmpeg, 然后通过pip安装  
    `pip install av`
- gluonfr  
    `pip install git+https://github.com/THUFutureLab/gluon-face.git@master`

## Pretrained Models
To be continued.

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


