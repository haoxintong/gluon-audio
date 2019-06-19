Install
===================================


Requirements
------------

mxnet-1.5.0+, gluonfr, av, librosa, …

音频库的选择主要考虑数据读取速度,
训练过程中音频的解码相比图像解码会消耗更多时间,
实际测试librosa从磁盘加载一个aac编码的短音频 耗时是pyav的8倍左右.

-  librosa

.. code:: shell

   pip install librosa

-  ffmpeg

.. code:: shell

    # 下载ffmpeg源码, 进入根目录
    ./configure --extra-cflags=-fPIC --enable-shared
    make -j
    sudo make install

-  pyav, 需要先安装ffmpeg

.. code:: shell

   pip install av

-  gluonfr

.. code:: shell

   pip install git+https://github.com/THUFutureLab/gluon-face.git@master