#  #################################################################
# This file includes functions to perform the WMMSE algorithm [2].
# Codes have been tested successfully on Python 3.6.0 with Numpy 1.12.0 support.
#
# References: [1]

# version 1.0 -- September 2017. Written by REVENOL (REVENOL AT outlook.com)
#  #################################################################

import scipy.io as sio                     # import scipy.io for .mat file I/O
import numpy as np                         # import numpy
import function_wmmse_powercontrol as wf   # import our function file
import function_dnn_powercontrol as df     # import our function file



for K in [10, 20, 30]:
    # Problem Setup, K: number of users.
    print('Gaussian IC Case: K=%d' % K)

    # Load model from this path, ! Please modify this path !
    model_location = "./DNNmodel/model_%d.ckpt" % (K)

    # Generate Testing Data
    num_test = 1000     # number of testing samples
    X, Y, wmmsetime = wf.generate_Gaussian(K, num_test, seed=7)

    # Testing Deep Neural Networks
    dnntime = df.test(X, model_location, "Prediction_%d" % K , K * K, K, binary=1)
    print('wmmse time: %0.3f s, dnn time: %0.3f s, time speed up: %0.1f X' % (wmmsetime, dnntime, wmmsetime / dnntime))

    # Evaluate Performance of DNN and WMMSE
    H = np.reshape(X, (K, K, X.shape[1]), order="F")
    NNVbb = sio.loadmat('Prediction_%d.mat' % K)['pred']
    wf.perf_eval(H, Y, NNVbb, K)
