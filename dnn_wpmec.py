#  #################################################################
#  This file contains the DNN code for our works on Wireless-powered Mobile Edge Computing [1], which includes training and testing functions.
#
#  References:
#  [1] Suzhi Bi, Liang Huang, Shengli Zhang, and Ying-jun Angela Zhang, Deep Neural Network for Computation Rate Maximization in Wireless Powered Mobile-Edge Computing Systems, submitted to IEEE Wireless Communications Letters.
#
# version 1.0 -- January 2018. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################

from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.io as sio
import time


def DNN_train(net,X_train,Y_train,X_valid,Y_valid,model_location,tensorboard_sw=0,export_weight_biase_sw=0,regularizer=0.001,training_epochs=100, batch_size=100, LR= 0.001,in_keep=0.95,hi_keep=0.95,LRdecay=0):

    n_input = X_train.shape[1]                          # input size
    n_output = Y_train.shape[1]                         # output size
    num_train = X_train.shape[0]
    num_valid = X_valid.shape[0]
    print('train: %d ' % num_train, 'validation: %d ' % num_valid)

    if tensorboard_sw==1:
        with tf.name_scope('inputs'):
            x = tf.placeholder("float", [None, n_input])
            y = tf.placeholder("float", [None, n_output])
        is_train = tf.placeholder("bool")

        learning_rate = tf.placeholder(tf.float32, shape=[])
        tf.summary.scalar('learning_rate',learning_rate)
        total_batch = int((X_train.shape[0]+X_valid.shape[0]) / batch_size)
        input_keep_prob = tf.placeholder(tf.float32)
        hidden_keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope('weights'):
            weights=[tf.Variable(tf.truncated_normal([n_input, net[0]]) / np.sqrt(n_input))]
            tf.summary.histogram('weight_1',weights[0])
            for i in range(len(net)-1):
                weights.append(tf.Variable(tf.truncated_normal([net[i], net[i+1]]) / np.sqrt(net[i])))
                tf.summary.histogram('weight_%d'%(i+2),weights[i+1])
            weights.append(tf.Variable(tf.truncated_normal([net[len(net)-1], n_output]) / net[i]))
            tf.summary.histogram('weight_%d'%(len(net)+1),weights[len(net)-1])

        with tf.name_scope('biases'):
            biases=[tf.Variable(tf.ones([net[0]]) * 0.05)]
            tf.summary.histogram('biase_1',biases[0])
            for i in range(len(net)-1):
                biases.append(tf.Variable(tf.ones([net[i+1]]) * 0.05))
                tf.summary.histogram('biase_%d'%(i+2),biases[i+1])
            biases.append(tf.Variable(tf.ones([n_output]) * 0.05))
            tf.summary.histogram('biase_%d'%(len(net)+1),biases[len(net)-1])

        with tf.name_scope('layers'):
            x1 = tf.nn.dropout(x, input_keep_prob)
            for i in range(len(net)):
                    x1 = tf.add(tf.matmul(x1, weights[i]), biases[i])   # x = wx+b
                    x1 = tf.nn.relu(x1)                                 # x = max(0, x)
                    x1 = tf.nn.dropout(x1, hidden_keep_prob)            # dropout layer
            pred = tf.matmul(x1, weights[len(net)]) + biases[len(net)]

        with tf.name_scope('loss'):
            if regularizer==0:
                cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels =y, logits = pred))
            else:
                cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels =y, logits = pred))
                for i in range(len(weights)):
                    cost=cost+tf.contrib.layers.l2_regularizer(regularizer)(weights[i])
            tf.summary.histogram('loss',cost)
    else:

        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_output])
        is_train = tf.placeholder("bool")

        learning_rate = tf.placeholder(tf.float32, shape=[])
        total_batch = int((X_train.shape[0]+X_valid.shape[0]) / batch_size)
        input_keep_prob = tf.placeholder(tf.float32)
        hidden_keep_prob = tf.placeholder(tf.float32)

        weights=[tf.Variable(tf.truncated_normal([n_input, net[0]]) / np.sqrt(n_input))]
        for i in range(len(net)-1):
            weights.append(tf.Variable(tf.truncated_normal([net[i], net[i+1]]) / np.sqrt(net[i])))
        weights.append(tf.Variable(tf.truncated_normal([net[len(net)-1], n_output]) / net[i]))

        biases=[tf.Variable(tf.ones([net[0]]) * 0.05)]
        for i in range(len(net)-1):
            biases.append(tf.Variable(tf.ones([net[i+1]]) * 0.05))
        biases.append(tf.Variable(tf.ones([n_output]) * 0.05))

        x1 = tf.nn.dropout(x, input_keep_prob)
        for i in range(len(net)):
                x1 = tf.add(tf.matmul(x1, weights[i]), biases[i])   # x = wx+b
                x1 = tf.nn.relu(x1)                                 # x = max(0, x)
                x1 = tf.nn.dropout(x1, hidden_keep_prob)            # dropout layer
        pred = tf.matmul(x1, weights[len(net)]) + biases[len(net)]

        if regularizer==0:
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels =y, logits = pred))
        else:
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels =y, logits = pred))
            for i in range(len(weights)):
                cost=cost+tf.contrib.layers.l2_regularizer(regularizer)(weights[i])

    optimizer = tf.train.AdamOptimizer(learning_rate, 0.09).minimize(cost)

    prediction = tf.sigmoid(pred)
    predicted_class = tf.greater(prediction, 0.5)
    correct = tf.equal(predicted_class, tf.equal(y,1.0))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    if tensorboard_sw==1:
        tf.summary.scalar('accuracy',accuracy)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        start_time = time.time()
        if tensorboard_sw==1:
            merged1 = tf.summary.merge_all()
            writer=tf.summary.FileWriter("./logs",sess.graph)

        for epoch in range(training_epochs):
            for i in range(total_batch):
                idx = np.random.randint(num_train,size=batch_size)
                if LRdecay==1:
                    _, c = sess.run([optimizer, cost], feed_dict={x: X_train[idx, :], y: Y_train[idx, :],
                                                                  input_keep_prob: in_keep, hidden_keep_prob: hi_keep,
                                                                  learning_rate: LR/(epoch+1), is_train: True})
                elif LRdecay==0:
                    _, c = sess.run([optimizer, cost], feed_dict={x: X_train[idx, :], y: Y_train[idx, :],
                                                                      input_keep_prob: in_keep, hidden_keep_prob: hi_keep,
                                                                      learning_rate: LR, is_train: True})

            if tensorboard_sw==1:
                result=sess.run(merged1, feed_dict={x: X_train[idx, :], y: Y_train[idx, :],
                                                                      input_keep_prob: in_keep, hidden_keep_prob: hi_keep,
                                                                      learning_rate: LR/(epoch+1), is_train: False})
                writer.add_summary(result,epoch)


            if epoch%10==0:
                accu, y_valid = sess.run([accuracy, pred], feed_dict={x: X_valid, y: Y_valid, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False})

                print('epoch:%d, '%epoch, 'train:%0.4f'%accuracy.eval({x: X_train, y: Y_train, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}), 'tcost:%0.4f'%cost.eval({x: X_train, y: Y_train, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}),'validation:%0.4f:'%accuracy.eval({x: X_valid, y: Y_valid, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}),'vcost:%0.4f:'%cost.eval({x: X_valid, y: Y_valid, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}))
        print('epoch:%d, '%training_epochs, 'train:%0.4f'%accuracy.eval({x: X_train, y: Y_train, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}), 'tcost:%0.4f'%cost.eval({x: X_train, y: Y_train, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}),'validation:%0.4f:'%accuracy.eval({x: X_valid, y: Y_valid, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}),'vcost:%0.4f:'%cost.eval({x: X_valid, y: Y_valid, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}))

        if export_weight_biase_sw==1:
            WB={}
            WB.setdefault('weight_1',sess.run(tf.Print(weights[0],[n_input ,net[0]])))
            for a in range(len(net)-1):
                WB.setdefault('weight_%d'%(a+2),sess.run(tf.Print(weights[a+1],[net[a] ,net[a+1]])))
            WB.setdefault('weight_%d'%(len(net)+1),sess.run(tf.Print(weights[len(net)],[net[(len(net)-1)] ,n_output])))

            WB.setdefault('biase_1',sess.run(tf.Print(biases[0],[net[0]])))
            for a in range(len(net)-1):
                WB.setdefault('biase_%d'%(a+2),sess.run(tf.Print(biases[a+1],[net[a+1]])))
            WB.setdefault('biase_%d'%(len(net)+1),sess.run(tf.Print(biases[len(net)],[n_output])))

            sio.savemat('./data/weights_biases',WB)

        print("Training Time: %0.2f s" % (time.time() - start_time))

        saver.save(sess, model_location)

    return 0

# Functions for deep neural network testing
def DNN_test(net,X_test, Y_test, gain, model_location, save_name,tensorboard_sw=0, binary=0):
    tf.reset_default_graph()
    n_input = X_test.shape[1]                          # input size
    n_output = Y_test.shape[1]                         # output size
    with tf.name_scope('inputs'):
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_output])
    is_train = tf.placeholder("bool")

    input_keep_prob = tf.placeholder(tf.float32)
    hidden_keep_prob = tf.placeholder(tf.float32)

    if tensorboard_sw==1:

        with tf.name_scope('weights'):
            weights=[tf.Variable(tf.truncated_normal([n_input, net[0]]) / np.sqrt(n_input))]
            tf.summary.histogram('weight_1',weights[0])
            for i in range(len(net)-1):
                weights.append(tf.Variable(tf.truncated_normal([net[i], net[i+1]]) / np.sqrt(net[i])))
                tf.summary.histogram('weight_%d'%(i+2),weights[i+1])
            weights.append(tf.Variable(tf.truncated_normal([net[len(net)-1], n_output]) / net[i]))
            tf.summary.histogram('weight_%d'%(len(net)+1),weights[len(net)-1])

        with tf.name_scope('biases'):
            biases=[tf.Variable(tf.ones([net[0]]) * 0.05)]
            tf.summary.histogram('biase_1',biases[0])
            for i in range(len(net)-1):
                biases.append(tf.Variable(tf.ones([net[i+1]]) * 0.05))
                tf.summary.histogram('biase_%d'%(i+2),biases[i+1])
            biases.append(tf.Variable(tf.ones([n_output]) * 0.05))
            tf.summary.histogram('biase_%d'%(len(net)+1),biases[len(net)-1])

        with tf.name_scope('layers'):
            x1 = tf.nn.dropout(x, input_keep_prob)
            for i in range(len(net)):
                    x1 = tf.add(tf.matmul(x1, weights[i]), biases[i])   # x = wx+b
                    x1 = tf.nn.relu(x1)                                 # x = max(0, x)
                    x1 = tf.nn.dropout(x1, hidden_keep_prob)            # dropout layer
            pred = tf.matmul(x1, weights[len(net)]) + biases[len(net)]

    else:

        weights=[tf.Variable(tf.truncated_normal([n_input, net[0]]) / np.sqrt(n_input))]
        for i in range(len(net)-1):
            weights.append(tf.Variable(tf.truncated_normal([net[i], net[i+1]]) / np.sqrt(net[i])))
        weights.append(tf.Variable(tf.truncated_normal([net[len(net)-1], n_output]) / net[i]))


        biases=[tf.Variable(tf.ones([net[0]]) * 0.05)]
        for i in range(len(net)-1):
            biases.append(tf.Variable(tf.ones([net[i+1]]) * 0.05))
        biases.append(tf.Variable(tf.ones([n_output]) * 0.05))


        x1 = tf.nn.dropout(x, input_keep_prob)
        for i in range(len(net)):
                x1 = tf.add(tf.matmul(x1, weights[i]), biases[i])   # x = wx+b
                x1 = tf.nn.relu(x1)                                 # x = max(0, x)
                x1 = tf.nn.dropout(x1, hidden_keep_prob)            # dropout layer
        pred = tf.matmul(x1, weights[len(net)]) + biases[len(net)]

    prediction = tf.sigmoid(pred)
    predicted_class = tf.greater(prediction, 0.5)
    correct = tf.equal(predicted_class, tf.equal(y,1.0))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_location)

        start_time = time.time()
        y_pred = sess.run(pred, feed_dict={x: X_test, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False})

        testtime = time.time() - start_time

        if binary==1:
            print('Test Accuracy:', accuracy.eval({x: X_test, y: Y_test, input_keep_prob: 1, hidden_keep_prob: 1, is_train: False}))

            y_pred = tf.sigmoid(y_pred)
            y_pred = tf.greater(y_pred, 0.5)
            y_pred = tf.cast(y_pred, tf.int32)

            y_pred = y_pred.eval()

        sio.savemat(save_name, {'input_h': X_test/10000000,'output_mode':Y_test,'output_mode_pred': y_pred,'output_obj':gain})
    return testtime, y_pred
