''' Skeleton-based gait index estimation with LSTMs (ICIS, IEEE 2018)
    BSD 2-Clause "Simplified" License
    Author: Trong-Nguyen Nguyen'''

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from LSTM_AE import *
from utils import *

save_result = False

'''=================== DATASET ======================'''
dataset = 'dataset/DIRO_skeletons.npz'
loaded = np.load(dataset)
skels = loaded['data']
separation = loaded['split']

n_full_joint = skels.shape[-1]//3
assert n_full_joint == 25

n_gait = skels.shape[1]

training_subjects = np.where(separation == 'train')[0]
test_subjects = np.where(separation == 'test')[0]
n_test_subject = len(test_subjects)

joints = [3, 20, 4, 8, 5, 9, 7, 11, 0, 12, 16, 13, 17, 14, 18, 15, 19] # selected joints
n_joint = len(joints)

seg_len = 12

'''==================== NETWORK ====================='''
tf.reset_default_graph()
tf.set_random_seed(2018)
np.random.seed(2018)

# constants
batch_num = 50
hidden_num = 256
step_num = seg_len
elem_num = len(joints)
epoch_num = 400//4

# load data
train_data_X, train_data_Y, train_data_Z, \
    test_data_normal_X, test_data_normal_Y, test_data_normal_Z, \
    test_data_abnormal_X, test_data_abnormal_Y, test_data_abnormal_Z = \
        loadskel(skels, training_subjects, joints, n_joint, half_seg_len = seg_len // 2, batchsize = batch_num)

# placeholder list
p_input = tf.placeholder(tf.float32, shape=(batch_num, step_num, elem_num))
print(p_input)
p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]

ae = LSTM_AE(hidden_num, p_input, step_num)
p_dropout = 0.5

def train_and_test(axis):
    if axis in ('x', 'X'):
        train_data = train_data_X
        test_data_normal = test_data_normal_X
        test_data_abnormal = test_data_abnormal_X
        weight_file_name = 'weights_X.txt'
    elif axis in ('y', 'Y'):
        train_data = train_data_Y
        test_data_normal = test_data_normal_Y
        test_data_abnormal = test_data_abnormal_Y
        weight_file_name = 'weights_Y.txt'
    elif axis in ('z', 'Z'):
        train_data = train_data_Z
        test_data_normal = test_data_normal_Z
        test_data_abnormal = test_data_abnormal_Z
        weight_file_name = 'weights_Z.txt'
    else:
        print('unknown axis')
        return None, None, None
    print('Training axis', axis, 'with', epoch_num, 'epochs...')
    losses = np.array([])
    err_train = np.zeros(train_data.shape[0])
    err_test_normal = np.zeros(test_data_normal.shape[0])
    err_test_abnormal = np.zeros(test_data_abnormal.shape[0])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch_num):
            for idx in range(train_data.shape[0]//batch_num):
                batch = train_data[idx*batch_num:(idx+1)*batch_num]
                (loss_val, _) = sess.run([ae.loss, ae.train], {p_input: batch, ae.prob: p_dropout})
                losses = np.append(losses, loss_val)
            
        #get cell's weights
        if save_result:
            enc_weights =  ae.get_enc_cell() # size: (17 + 256) x (256 * 4) -> i = input_gate, j = new_input, f = forget_gate, o = output_gate
            write_weights_to_file(weight_file_name, enc_weights)
            print('finish writing encoder weights to ' + weight_file_name)
        #calc err of train and test data
        for idx in range(train_data.shape[0]//batch_num):
            (input_, output_) = sess.run([ae.input_, ae.output_], {p_input: train_data[idx*batch_num:(idx+1)*batch_num]})
            for i in range(batch_num):
                err_train[idx * batch_num + i] = np.mean((input_[i] - output_[i])**2)
        for idx in range(test_data_normal.shape[0]//batch_num):
            (input_, output_) = sess.run([ae.input_, ae.output_], {p_input: test_data_normal[idx*batch_num:(idx+1)*batch_num]})
            for i in range(batch_num):
                err_test_normal[idx * batch_num + i] = np.mean((input_[i] - output_[i])**2)
        for idx in range(test_data_abnormal.shape[0]//batch_num):
            (input_, output_) = sess.run([ae.input_, ae.output_], {p_input: test_data_abnormal[idx*batch_num:(idx+1)*batch_num]})
            for i in range(batch_num):
                err_test_abnormal[idx * batch_num + i] = np.mean((input_[i] - output_[i])**2)

        print(input_.shape)
        print(output_.shape)
        return err_train, err_test_normal, err_test_abnormal, losses

err_train_X, err_test_normal_X, err_test_abnormal_X, losses_train_X = train_and_test('X')
print('X-axis')
auc = calc_AUC(err_test_abnormal_X, err_test_normal_X)

err_train_Y, err_test_normal_Y, err_test_abnormal_Y, losses_train_Y = train_and_test('Y')
print('Y-axis')
auc = calc_AUC(err_test_abnormal_Y, err_test_normal_Y)

err_train_Z, err_test_normal_Z, err_test_abnormal_Z, losses_train_Z = train_and_test('Z')
print('Z-axis')
auc = calc_AUC(err_test_abnormal_Z, err_test_normal_Z)

'''==================== ASSESSMENT ====================='''

'''simple summation'''
print('=== summation ===')
# per-segment
sum_abnormal = err_test_abnormal_X + err_test_abnormal_Y + err_test_abnormal_Z
sum_normal = err_test_normal_X + err_test_normal_Y + err_test_normal_Z
print('per-segment (non-weighted sum)')
auc = calc_AUC(sum_abnormal, sum_normal)

# per-sequence
sum_abnormal = np.mean(np.split(sum_abnormal,n_test_subject * (n_gait - 1)), axis = 1)
sum_normal = np.mean(np.split(sum_normal,n_test_subject), axis = 1)
print('per-sequence (non-weighted sum)')
auc = calc_AUC(sum_abnormal, sum_normal)

'''weighted combination'''
err_mean_train_X = np.mean(err_train_X)
err_mean_train_Y = np.mean(err_train_Y)
err_mean_train_Z = np.mean(err_train_Z)
weight_X = (err_mean_train_X + err_mean_train_Y + err_mean_train_Z) / err_mean_train_X
weight_Y = (err_mean_train_X + err_mean_train_Y + err_mean_train_Z) / err_mean_train_Y
weight_Z = (err_mean_train_X + err_mean_train_Y + err_mean_train_Z) / err_mean_train_Z
# per-segment
sum_abnormal = weight_X * err_test_abnormal_X + weight_Y * err_test_abnormal_Y + weight_Z * err_test_abnormal_Z
sum_normal = weight_X * err_test_normal_X + weight_Y * err_test_normal_Y + weight_Z * err_test_normal_Z
print('per-segment (weighted sum)')
auc = calc_AUC(sum_abnormal, sum_normal)

# per-sequence
sum_abnormal = np.mean(np.split(sum_abnormal,n_test_subject * (n_gait - 1)), axis = 1)
sum_normal = np.mean(np.split(sum_normal,n_test_subject), axis = 1)
print('per-sequence (weighted sum)')
auc = calc_AUC(sum_abnormal, sum_normal)
