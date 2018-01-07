#  #################################################################
#  Python code to reproduce our works on Wireless-powered Mobile Edge Computing [1], which uses the wireless channel gains as the input and the binary computing mode selection results as the output of a deep neural network (DNN).
#
#  This file contains the main code to train and test the DNN. It loads the training samples saved in ./data/data_#.mat, splits the samples into three parts (training, validation, and testing data constitutes 60%, 20% and 20%), trains the DNN with training and validation samples, and finally tests the DNN with test data.
#
#  Input: ./data/data_#.mat
#    Data samples are generated according to the CD method presented in [2]. THere are 30,000 samples saved in each ./data/data_#.mat, where # is the user number. Each data sample includes
#  -----------------------------------------------------------------
#  |       wireless channel gain           |    input_h            |
#  -----------------------------------------------------------------
#  |       computing mode selection        |    output_mode        |
#  -----------------------------------------------------------------
#  |       energy broadcasting parameter   |    output_a           |
#  -----------------------------------------------------------------
#  |     transmit time of wireless device  |    output_tau         |
#  -----------------------------------------------------------------
#  |      weighted sum computation rate    |    output_obj         |
#  -----------------------------------------------------------------
#
#  Output:
#    - Training Time: the time cost to train 18,000 independent data samples
#    - Testing Time: the time cost to compute predicted 6,000 computing mode
#    - Test Accuracy: the accuracy of the predicted mode selection. Please note that the mode selection accuracy is different from computation rate accuracy, since two different computing modes may leads to similar weighted sum computation rates. From our experience, the accuracy of weighted sum computation rate (evaluated as DNN/CD) is higher than the accuracy of computing mode selection.
#    - ./data/weights_biases.mat: parameters of the trained DNN, which are used to re-produce this trained DNN in MATLAB.
#    - ./data/Prediction_#.mat
#    Besides the test data samples, it also includes the predicted mode selection. Given DNN-predicted mode selection, the corresponding optimal weighted sum computation rate can be computed by solving (P1) in [1], which achieves over 99.9% of the CD method [2].
#  -----------------------------------------------------------------
#  |       wireless channel gain           |    input_h            |
#  -----------------------------------------------------------------
#  |       computing mode selection        |    output_mode        |
#  -----------------------------------------------------------------
#  |       DNN-predicted mode selection    |    output_mode_pred   |
#  -----------------------------------------------------------------
#  |      weighted sum computation rate    |    output_obj         |
#  -----------------------------------------------------------------
#
#  References:
#  [1] Suzhi Bi, Liang Huang, Shengli Zhang, and Ying-jun Angela Zhang, Deep Neural Network for Computation Rate Maximization in Wireless Powered Mobile-Edge Computing Systems, submitted to IEEE Wireless Communications Letters.
#  [2] S. Bi and Y. J. Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” submitted for publication, available on-line at arxiv.org/abs/1708.08810.
#
# version 1.0 -- January 2018. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################

import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import dnn_wpmec as dnn     # import our function file


K = 30                     # number of users


# Load data
channel = sio.loadmat('./data/data_%d' %K)['input_h']
mode = sio.loadmat('./data/data_%d' %K)['output_mode']
harvesting_time = sio.loadmat('./data/data_%d' %K)['output_a']
offloading_time = sio.loadmat('./data/data_%d' %K)['output_tau']
gain = sio.loadmat('./data/data_%d' %K)['output_obj']

# pre-process data
#offloading_time[offloading_time == 0]=-1
offloading_time = mode
channel = channel * 10000000

# train:validation:test = 60:20:20
split_idx = [int(.6*len(channel)), int(.8*len(channel))]
X_train, X_valid, X_test = np.split(channel, split_idx)
mode_train, mode_valid, mode_test = np.split(mode, split_idx)
harvesting_time_train, harvesting_time_valid, harvesting_time_test = np.split(harvesting_time, split_idx)
Y_train, Y_valid, Y_test = np.split(offloading_time, split_idx)
gain_train, gain_valid, gain_test = np.split(gain, split_idx)

# Save & Load model from this path
model_location = "./DNNmodel/model_demo.ckpt"
save_name="./data/Prediction_%d" % K

#open the tensorboard or not
tensorboard_sw=0
#export the weights and biases or not
export_weight_biase_sw=1

#hyper-parameter
net=[120,80]
training_epochs=130
regularizer=0.0005
batch_size=100
LR = 0.001
in_keep=1
hi_keep=1
LRdecay=1



print('Case: K=%d, Total Samples: %d, Total Iterations: %d, layers:%d\n'%(K, len(channel), training_epochs,len(net)))

# Train the deep neural network
print('train DNN ...')
dnn.DNN_train(net,X_train, Y_train, X_valid, Y_valid,model_location,tensorboard_sw,export_weight_biase_sw,regularizer,training_epochs,batch_size,LR,in_keep,hi_keep,LRdecay)

# Testing Deep Neural Networks
dnntime, Y_pred = dnn.DNN_test(net,X_test, Y_test, gain_test, model_location,save_name, tensorboard_sw,binary=1)
print('Testing Time: %0.3f s' % (dnntime))
