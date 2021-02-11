import tensorflow as tf
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import tensorflow.python.framework.dtypes

from numpy import vstack
#from sklearn.datasets import fetch_california_housing
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from feature_reduction import *
from autoencoder import *

import pandas as pd
import autokeras as ak
import timeit
import time
import pdb

from configuration import args
from prepare_data import generate_datasets, normalize
from Visualization_Bayesian import *

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import timeit
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def csr2dense(X_train_sparse):
    x_data = np.array(X_train_sparse)
    X_batch = []
    for i in range(len(x_data)):
        x = np.array(x_data[i].todense())
        X_batch.append(x)
    X_batch = np.array(X_batch)
    return X_batch

def coo_to_tensor(x_data):
    X_tf_ind_array = []
    X_tf_val_array = []
    for i in range(len(x_data)):
        X_tf_ind = tf.SparseTensor(indices=np.column_stack((x_data[i].row, x_data[i].col)), values=x_data[i].col, dense_shape=x_data[i].shape)
        X_tf_val = tf.SparseTensor(indices=np.column_stack((x_data[i].row, x_data[i].col)), values=x_data[i].data, dense_shape=x_data[i].shape)
        X_tf_ind_array.append(X_tf_ind)
        X_tf_val_array.append(X_tf_val)
    return X_tf_ind_array, X_tf_val_array

def embedding_lookup_sparse(X_tf_ind_array, X_tf_val_array, V, W):
    embedded_matrix = []
    timeconsume = 0.00
    for i in range(len(X_tf_ind_array)):
        part_1=time.clock()
        result = tf.nn.embedding_lookup_sparse(V, X_tf_ind_array[i], X_tf_val_array[i], combiner='sum')
        result += W
        part_2=time.clock()
        timeconsume += part_2 - part_1
        embedded_matrix.append(tf.reshape(result, [-1]))
    embedded_matrix = np.array(embedded_matrix)
    # print("\tFeature Reduction Time:" + str([round(float(timeconsume), 5)]))
    # print("\tAvg. Feature Reduction Time:" + str([round(float(timeconsume)/len(X_tf_ind_array), 5)]))
    return embedded_matrix, float(timeconsume)/len(X_tf_ind_array)


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # import pdb; pdb.set_trace()
    # get the inference dataset
    read_start = time.clock()
    X_train_sparse, Y_train_data= generate_datasets()
    read_end = time.clock()
    read_timeconsume = read_end - read_start
    read_time = float(read_timeconsume)/len(X_train_sparse)
    # print("\tData Read Time:" + str([round(float(read_timeconsume), 5)]))
    # print("\tAvg. Data Read Time:" + str([round(float(read_timeconsume)/len(X_train_sparse), 5)]))

    autoencoder_path = "./"+ args.benchmark +"/autoencoder"
    encoder = tf.keras.models.load_model(autoencoder_path)
    tf.keras.utils.plot_model(
        encoder, to_file=args.save_bayesian_path + 'autoencoder.png', show_shapes=False, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )
    # import pdb; pdb.set_trace()
    # get the embedding kernel matrix and do feature reduction on sparse matrix
    W = encoder.layers[0].get_weights()[1] #biases
    V = encoder.layers[0].get_weights()[0] #wights

    if (args.benchmark=='AMG'):
        X_tf_ind_array, X_tf_val_array = coo_to_tensor(X_train_sparse)
        X_encoding, fea_time = embedding_lookup_sparse(X_tf_ind_array, X_tf_val_array, V, W)
    elif (args.benchmark=='CG'):
        X_train_sparse = csr2dense(X_train_sparse)
        read_start = time.clock()
        X_encoding = np.array(np.dot(X_train_sparse,V)+W)
        read_end = time.clock()
        fea_timeconsume = read_end - read_start
        fea_time = float(fea_timeconsume)/len(X_encoding)
    elif (args.benchmark=='MG' or args.benchmark=='Lagos_fine' or args.benchmark=='Lagos_coarse'):
        read_start = time.clock()
        X_encoding = np.array(np.dot(X_train_sparse,V)+W)
        read_end = time.clock()
        fea_timeconsume = read_end - read_start
        fea_time = float(fea_timeconsume)/len(X_encoding)

    split_index = int(math.floor(len(X_encoding)* args.TRAIN_SET_RATIO / 100.0))
    assert (split_index >= 0 and split_index <= len(X_encoding))
    x_train = X_encoding[:split_index]
    x_test = X_encoding[split_index:]
    y_train = np.array(Y_train_data[:split_index])
    if (args.benchmark=='CG'):
        y_test = np.array(Y_train_data[split_index:args.sample_size])
    else:
        y_test = np.array(Y_train_data[split_index:])


    #ml_loss, initial_history, final_history = Model_search(x_train, y_train, x_test, y_test)
    # Autokeras searched model
    model_path =  "./"+ args.benchmark +"/best_model"
    model = tf.keras.models.load_model(model_path)
    tf.keras.utils.plot_model(
        model, to_file=args.save_bayesian_path + 'model.png', show_shapes=False, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )
    model.summary()
    #retrain the model
    model.fit(x_train, y_train, epochs = 500, validation_data = (x_test, y_test), verbose=1)

    # tf.profiler.experimental.start('logdir')
    start = time.clock()
    y_predict = model.predict(x_test)
    # tf.profiler.experimental.stop()
    end = time.clock()
    timeconsume = end - start
    # print("\tModel Inference Time:" + str([round(float(timeconsume), 5)]))
    # print("\tAvg. Model Inference Time:" + str([round(float(timeconsume)/len(x_test), 5)]))
    infer_time = float(timeconsume)/len(x_test)
    loss = np.mean(np.abs(y_predict-y_test))
    print(loss)

    time_record = np.hstack([read_time, fea_time, infer_time])
    # np.savetxt(args.save_bayesian_path + "x_test.txt", np.array(x_test), fmt='%.16f',
    #            delimiter=',')
    np.savetxt(args.save_bayesian_path + "y_test.txt", np.array(y_test), fmt='%.16f',
               delimiter=',')
    np.savetxt(args.save_bayesian_path + "y_prediction.txt", np.array(y_predict), fmt='%.16f',
               delimiter=',')
    np.savetxt(args.save_bayesian_path + "time_record.txt", np.array(time_record), fmt='%.16f',
               delimiter=',')
    # print(y_predict)
