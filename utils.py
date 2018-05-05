import numpy as np
from sklearn.metrics import roc_auc_score

# old shape: nFrame * nJoint => new shape: k * (2 * half_seg_len) * nJoint
def segmentseq(input_data, half_seg_len, batchsize = 0, overlapping = True):
    if overlapping:
        k = input_data.shape[0] // half_seg_len - 1
    else:
        k = input_data.shape[0]//(2*half_seg_len)
    ret_data = np.zeros([k, 2 * half_seg_len, input_data.shape[1]])
    for i in range(k):
        if overlapping:
            ret_data[i] = input_data[i * half_seg_len : (i+2) * half_seg_len]
        else:
            ret_data[i] = input_data[i * (2 * half_seg_len) : (i + 1) * (2 * half_seg_len)]
    if batchsize > 0 and k % batchsize != 0:
        missing = batchsize - k % batchsize
        ret_data = np.concatenate((ret_data, ret_data[0:missing]), axis=0)
    return ret_data

# scaling skeleton so that X,Y,Z in range [0, 1], process on each row (as 1 frame)
def normalizecoordXYZ(input_data, n_full_joint = 25):
    ret_data = np.copy(input_data)
    for i in range(input_data.shape[0]):
        skel = input_data[i,:]
        tmp = np.zeros(n_full_joint)
        #
        max_vals = [-float("inf"), -float("inf"), -float("inf")]
        min_vals = [float("inf"), float("inf"), float("inf")]
        #
        for j in range(len(skel)):
            min_vals[j%3] = skel[j] if min_vals[j%3] > skel[j] else min_vals[j%3]
            max_vals[j%3] = skel[j] if max_vals[j%3] < skel[j] else max_vals[j%3]
        for j in range(len(skel)):
            ret_data[i,j] = (skel[j] - min_vals[j%3])/(max_vals[j%3] - min_vals[j%3])
    return ret_data

# input_data: skeletons with normalized range
def getjoints(input_data, joints, n_joint = 17):
    dataX = np.zeros([input_data.shape[0], n_joint])
    dataY = np.zeros([input_data.shape[0], n_joint])
    dataZ = np.zeros([input_data.shape[0], n_joint])
    for i in range(n_joint):
        dataX[:,i] = input_data[:,joints[i]*3]
        dataY[:,i] = input_data[:,joints[i]*3+1]
        dataZ[:,i] = input_data[:,joints[i]*3+2]
    return dataX, dataY, dataZ

def loadskel(skel_data, training_subjects, joints, n_joint, half_seg_len, batchsize = 0):
    # training data: overlapping => increase number of samples
    train_data_X = np.array([]).reshape(0, 2 * half_seg_len, n_joint)
    train_data_Y = np.array([]).reshape(0, 2 * half_seg_len, n_joint)
    train_data_Z = np.array([]).reshape(0, 2 * half_seg_len, n_joint)
    # test data: non-overlapping
    test_data_normal_X = np.array([]).reshape(0, 2 * half_seg_len, n_joint)
    test_data_normal_Y = np.array([]).reshape(0, 2 * half_seg_len, n_joint)
    test_data_normal_Z = np.array([]).reshape(0, 2 * half_seg_len, n_joint)
    test_data_abnormal_X = np.array([]).reshape(0, 2 * half_seg_len, n_joint)
    test_data_abnormal_Y = np.array([]).reshape(0, 2 * half_seg_len, n_joint)
    test_data_abnormal_Z = np.array([]).reshape(0, 2 * half_seg_len, n_joint)   
    #
    n_subject, n_gait = skel_data.shape[:2]
    for i in range(n_subject):
        for j in range(n_gait):
            if i in training_subjects and j > 0:
                continue # do not use abnormal gaits of training subjects
            #
            data = skel_data[i,j]
            data = normalizecoordXYZ(data)
            dataX, dataY, dataZ = getjoints(data, joints, n_joint = n_joint)
            dataX = segmentseq(dataX, half_seg_len, batchsize, overlapping = i in training_subjects)
            dataY = segmentseq(dataY, half_seg_len, batchsize, overlapping = i in training_subjects)
            dataZ = segmentseq(dataZ, half_seg_len, batchsize, overlapping = i in training_subjects)
            if i in training_subjects:                
                train_data_X = np.concatenate((train_data_X, dataX), axis=0)
                train_data_Y = np.concatenate((train_data_Y, dataY), axis=0)
                train_data_Z = np.concatenate((train_data_Z, dataZ), axis=0)
            else:
                if j == 0:
                    test_data_normal_X = np.concatenate((test_data_normal_X, dataX), axis=0)
                    test_data_normal_Y = np.concatenate((test_data_normal_Y, dataY), axis=0)
                    test_data_normal_Z = np.concatenate((test_data_normal_Z, dataZ), axis=0)
                else:
                    test_data_abnormal_X = np.concatenate((test_data_abnormal_X, dataX), axis=0)
                    test_data_abnormal_Y = np.concatenate((test_data_abnormal_Y, dataY), axis=0)
                    test_data_abnormal_Z = np.concatenate((test_data_abnormal_Z, dataZ), axis=0)
    print('Finish loading data')
    print(train_data_X.shape)
    print(test_data_normal_X.shape)
    print(test_data_abnormal_X.shape)
    return train_data_X, train_data_Y, train_data_Z, \
            test_data_normal_X, test_data_normal_Y, test_data_normal_Z, \
            test_data_abnormal_X, test_data_abnormal_Y, test_data_abnormal_Z

def calc_AUC(seq_abnormal, seq_normal, print_auc = True):
    y_true = np.concatenate((np.ones(len(seq_abnormal)), np.zeros(len(seq_normal))), axis = 0)
    y_pred = np.concatenate((seq_abnormal, seq_normal), axis = 0)
    auc = roc_auc_score(y_true, y_pred)
    if print_auc:
        print('AUC = %.3f' % auc)
    return auc

def write_weights_to_file(filename, data):
    with open(filename, "a") as myfile:
        myfile.write(str(data.shape) + '\n[')
        for n_row in range(data.shape[0]):
            myfile.write('\n')
            for val in data[n_row,:]:
                myfile.write(str(val))
                myfile.write(' ')
            if n_row != data.shape[0] - 1:
                myfile.write(';...')
        myfile.write(']\n')
