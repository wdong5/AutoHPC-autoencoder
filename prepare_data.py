import tensorflow as tf
import pathlib
import numpy as np
import os
import os.path
import math
import random
import pdb

from configuration import args
from feature_reduction import Shannon_reduction, PCA_reduction
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

np.random.seed(1337)  # for reproducibility
#from parse_tfrecord import get_parsed_dataset

def get_sparse_matrix(x_data_dir, x_row_dir, x_col_dir):
    data_list = open(os.path.join(args.dataset_dir, x_data_dir), 'r')
    row_list = open(os.path.join(args.dataset_dir, x_row_dir), 'r')
    col_list = open(os.path.join(args.dataset_dir, x_col_dir), 'r')

    sparse_x = []
    x_size = 0
    for x_data_str in data_list.readlines():
        x_data = x_data_str.split()
        x_data = np.array([float(i) for i in x_data])
        x_indptr = row_list.readline().split()
        x_indices = col_list.readline().split()
        X = csr_matrix((x_data, x_indices, x_indptr), shape=(args.matrix_row, args.matrix_col))
        sparse_x.append(X)
        x_size = x_size+1
        if x_size == args.sample_size:
            break
    split_index = int(math.floor(x_size* args.TRAIN_SET_RATIO / 100.0))
    assert (split_index >= 0 and split_index <= x_size)
    train_list = sparse_x[:split_index]
    test_list = sparse_x[split_index:]
    return train_list, test_list

def get_sparse_files(A_data_dir, x_data_dir):
    A_data_dir = os.path.join(args.dataset_dir, A_data_dir)
    x_data_dir = os.path.join(args.dataset_dir, x_data_dir)
    A_data_files= os.listdir(A_data_dir)
    x_data_files= os.listdir(x_data_dir)
    A_data = []
    x_data = []
    #read matrix A as NN inputs
    for file in A_data_files:
        A_matrix_data = []
        if not os.path.isdir(file):
          data_list = open(os.path.join(A_data_dir,file),'r')
          row_col = data_list.readline().split()
          for A_data_str in data_list.readlines():
              # pdb.set_trace()
              A_ele = A_data_str.split()
              float(A_ele[2])
              A_matrix_data.append(A_ele)
          A_matrix_data = np.array(A_matrix_data)
          A_matrix = coo_matrix((A_matrix_data[:,2], (A_matrix_data[:,0], A_matrix_data[:,1])),shape=(int(row_col[1])+1, int(row_col[3])+1), dtype=np.float32)
        A_data.append(A_matrix)
        # pdb.set_trace()
    #read array x as NN single_output
    for file in x_data_files:
        x_array_data = []
        if not os.path.isdir(file):
          data_list = open(os.path.join(x_data_dir,file),'r')
          row_col = data_list.readline().split()
          for x_data_str in data_list.readlines():
              x_ele = x_data_str.split()
              # pdb.set_trace()
              x_array_data.append(float(x_ele[1]))
        x_data.append(x_array_data)
    x_data = np.array(x_data)
    # pdb.set_trace()
    #split data into training and validation sets
    split_index = int(math.floor(len(A_data) * args.TRAIN_SET_RATIO / 100.0))
    assert (split_index >= 0 and split_index <= len(A_data))
    # pdb.set_trace()
    train_list = A_data[:split_index]
    test_list = A_data[split_index:]
    train_x = x_data[:split_index]
    test_x  = x_data[split_index:]
    # pdb.set_trace()
    return  train_list, test_list, train_x, test_x

def get_the_length_of_dataset(dataset):
    count = 0
    for i in dataset:
        count += 1
    return count

def load_dataset(list_file):
    raw_im_list = np.loadtxt(os.path.join(args.dataset_dir, list_file))
    assert len(raw_im_list) > 0
    #random.shuffle(raw_im_list)
    split_index = int(math.floor(len(raw_im_list) * args.TRAIN_SET_RATIO / 100.0))
    assert (split_index >= 0 and split_index <= len(raw_im_list))
    train_list = raw_im_list[:split_index]
    test_list = raw_im_list[split_index:]
    return  train_list, test_list

def normalize(matrix):
    norm =np.linalg.norm(matrix)
    if norm == 0:
        return matrix
    boolArr = (matrix == 0)
    min = matrix.min()
    if min == 0:
        matrix_nonzero = matrix[matrix>0]
        min = matrix_nonzero.min()
    matrix_x = (matrix - min) / norm
    matrix_x[boolArr] = 0.0
    return matrix_x

def generate_datasets():
    if (args.benchmark=='MG'):
        train_value, test_value = get_sparse_matrix("cg_csr_value.txt", "cg_csr_row_ind.txt", "cg_csr_col_ind.txt")
        train_x, test_x = load_dataset("cg_x.txt")
    elif (args.benchmark=='AMG'):
        train_value, test_value, train_x, test_x  = get_sparse_files("IJ_A", "IJ_x")
    # if (args.preprocessing==None):
    #     train_value, test_value = get_sparse_matrix("cg_csr_value.txt", "cg_csr_row_ind.txt", "cg_csr_col_ind.txt")
    #     train_x, test_x = load_dataset("cg_x.txt")
    # elif (args.preprocessing=='stad'):
    #     train_value, test_value = get_sparse_matrix("cg_csr_value_stad.txt", "cg_csr_row_ind.txt", "cg_csr_col_ind.txt")
    #     train_x, test_x = load_dataset("cg_x_stad.txt")
    # elif (args.preprocessing=='norm'):
    #     train_value, test_value = get_sparse_matrix("cg_csr_value_norm.txt", "cg_csr_row_ind.txt", "cg_csr_col_ind.txt")
    #     train_x, test_x = load_dataset("cg_x_norm.txt")
    return train_value,  train_x, test_value, test_x
