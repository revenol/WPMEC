#  #################################################################
#  Python code to reproduce our works on DNN research for ****.
#
#  This file contains the whole process from data generation, training, testing to plotting
#  for 10 users' IC case, even though such process done on a small dataset of 25000 samples,
#  94% accuracy can still be easily attained in less than 100 iterations.
#
# This file includes functions to perform the WMMSE algorithm [2].
# Codes have been tested successfully on Python 3.6.0 with Numpy 1.12.0 support.
#
# References: [1]

# version 1.0 -- September 2017. Written by REVENOL (REVENOL AT outlook.com)
#  #################################################################

import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
#import matplotlib.pyplot as plt            # import matplotlib.pyplot for figure plotting
import function_dnn_resource_allocation as dnn     # import our function file


K = 30                     # number of users
#training_epochs = 100      # number of training epochs
trainseed = 0              # set random seed for training set
testseed = 7               # set random seed for test set

# Load data
Data = sio.loadmat('./data/data_%d' %K)

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
save_name="Prediction_%d" % K
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

print('train DNN ...')
#dnn.train(X_train, Y_train, X_valid, Y_valid, model_location, training_epochs=training_epochs, batch_size=100)
dnn.DNN_train(net,X_train, Y_train, X_valid, Y_valid,model_location,tensorboard_sw,export_weight_biase_sw,regularizer,training_epochs,batch_size,LR,in_keep,hi_keep,LRdecay)

# Testing Deep Neural Networks
dnntime, Y_pred = dnn.DNN_test(net,X_test, Y_test, gain_test, model_location,save_name, tensorboard_sw,binary=1)
print('dnn time: %0.3f s' % (dnntime))

#Y_error = Y_pred - Y_test
# # Evaluate Performance of DNN and WMMSE
# H = np.reshape(X, (K, K, X.shape[1]), order="F")
# NNVbb = sio.loadmat('Prediction_%d.mat' % K)['pred']
# wf.perf_eval(H, Y, NNVbb, K)
#
# # Plot figures
# train = sio.loadmat('MSETime_%d_%d_%d'%(K, 200, 10))['train']
# time = sio.loadmat('MSETime_%d_%d_%d'%(K, 200, 10))['time']
# val = sio.loadmat('MSETime_%d_%d_%d'%(K, 200, 10))['validation']
# plt.figure(0)
# plt.plot(time.T, val.T,label='validation')
# plt.plot(time.T, train.T,label='train')
# plt.legend(loc='upper right')
# plt.xlabel('time (seconds)')
# plt.ylabel('Mean Square Error')
# plt.savefig('MSE_train.eps', format='eps', dpi=1000)
# plt.show()
