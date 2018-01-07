# 如何安装 tensorflow

> Tensorflow 的官网提供了详细的[安装说明](https://www.tensorflow.org/install/)。本文介绍我们是如何在Windows操作系统上安装tensorflow。希望可以帮助tensorflow的新人快速上手。同时可以参考相关[博客](https://www.w3cschool.cn/tensorflow/tensorflow-lbqi2chw.html)。

1. 安装 Anaconda.

    从anaconda的[官网](https://www.anaconda.com/download/)免费下载软件并安装。我们建议选择Python 3.6的版本，根据自己的Windows版本选择是[32位软件](https://repo.continuum.io/archive/Anaconda3-4.4.0-Windows-x86.exe)还是[64位软件](https://repo.continuum.io/archive/Anaconda3-4.4.0-Windows-x86_64.exe)。关于anaconda的使用，包括相关软件 Anaconda Navigator， Jupyter notebook，和 Spyder等可以参考相关[博客](https://segmentfault.com/a/1190000011126204)。

2. 新建conda环境，命名为tensorflow，输入以下命令：

    `C:> conda create -n tensorflow python=3.6`

3. 激活新建的tensorflow环境，输入以下命令：
    ```
    C:> activate tensorflow

     (tensorflow)C:>  # 这里显示会更具系统不同而变化
    ```

4. 安装TensorFlow软件。可以选择CPU版本或者GPU版本。

    CPU版本：

    `(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow`

    GPU显卡安装：

    `(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow-gpu `

5. 测验安装。新建命令行，依次输入以下命令

    ```
    C:> activate tensorflow
    $ Python
    >>> import tensorflow as tf
    >>> hello = tf.constant('Hello, TensorFlow!')
    >>> sess = tf.Session()
    >>> print(sess.run(hello))
    ```
    如果TensorFlow安装成功，会得到以下输出结果

    `Hello, TensorFlow!`

1. 图形安装所需要的软件包，numpy 和 scipy ：
  - 右键，以管理员权限 打开 Anaconda Navigator
  - 点击左侧 Environments
  - 中间栏，选择 tensorflow 环境
  - 右侧栏，下拉菜单选择 Not Installed，默认选择为 installed，然后在'Channles','Update index...'右边的输入框输入所需要的安装包，例如 scipy，回车后下方表格会显示对应的scipy安装包
  - 左键勾选 scipy左边的方框，然后点击右下的 Apply 按钮，Anaconda Navigator 会开始安装相应的软件包

1. 图形界面运行我们的程序：
  - 右键，以管理员权限 打开 Anaconda Navigator
  - 点击左侧 Home，在 Applications on 右边的下拉菜单中，选择 tensorflow
  - 在下方的 spyder 软件，单击 install，会安装 spyder （一种IDE，初次使用需安装）
  - spyder 成功安装后，点击 Launch，进入 pyder 软件。打开 main.py 文件，点击 run 即可。
  *注意，每次运行 spyder 之前，都要先选择 tensorflow 环境，不然运行文件会提示 tensorflow 错误。*
