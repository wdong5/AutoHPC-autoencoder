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

# This is for CG data fetch
def get_CG_matrix(data_list, row_list, col_list):
    # data_list = open(os.path.join(args.dataset_dir, x_data_dir), 'r')
    # row_list = open(os.path.join(args.dataset_dir, x_row_dir), 'r')
    # col_list = open(os.path.join(args.dataset_dir, x_col_dir), 'r')
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
    return sparse_x

# This is for AMG data fetch
def get_AMG_files(A_data_dir, x_data_dir):
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
    return  A_data, x_data

# This is for MG data fetch
def get_MG_matrix(r_int_list, r_sol_list):
    r_array = []
    r_sol_array = []
    for r_int_str in r_int_list.readlines():
        r_int = r_int_str.split()
        r_data = np.array([float(i) for i in r_int])
        r_array.append(r_data)

    for r_sol_str in r_sol_list.readlines():
        r_sol = r_sol_str.split()
        r_sol_data = np.array([float(i) for i in r_sol])
        r_sol_array.append(r_sol_data)

    return r_array, r_sol_array

# This is for Lagos data fetch
def get_Lagos_files(A_data_dir):
    A_data_files= os.listdir(A_data_dir)
    A_data = []
    for file in A_data_files:
        A_matrix_data = []
        if not os.path.isdir(file):
          data_list = open(os.path.join(A_data_dir,file),'r')
          for i, A_data_str in enumerate(data_list):
              if (i>6):
                  A_ele = float(A_data_str)
                  A_matrix_data.append(A_ele)
          A_matrix_data = np.array(A_matrix_data)
        A_data.append(A_matrix_data)
    return A_data

def get_the_length_of_dataset(dataset):
    count = 0
    for i in dataset:
        count += 1
    return count

def load_dataset(list_file):
    raw_im_list = np.loadtxt(list_file)
    return  raw_im_list

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
    if (args.benchmark=='CG'):
        #CG dataset
        path = "/home/cc/AutoHPCnet-benchmark/NPB3.3-SER-C/CG/CG_dataset"
        cg_csr = open(os.path.join(path, "cg_csr_value.txt"), 'r')
        cg_csr_row = open(os.path.join(path, "cg_csr_row_ind.txt"), 'r')
        cg_csr_col = open(os.path.join(path, "cg_csr_col_ind.txt"), 'r')
        cg_x = open(os.path.join(path, "cg_x.txt"), 'r')
        train_value = get_CG_matrix(cg_csr, cg_csr_row, cg_csr_col)
        train_x= load_dataset(cg_x)

    elif (args.benchmark=='AMG'):
        path = "/home/cc/AutoHPCnet-benchmark/AMG/test"
        A_data_dir = os.path.join(path, "IJ_A")
        x_data_dir = os.path.join(path, "IJ_x")
        train_value, train_x = get_AMG_files(A_data_dir, x_data_dir)

    elif (args.benchmark=='MG'):
        path = "/home/cc/AutoHPCnet-benchmark/NPB3.3-SER-C/MG/MG_dataset"
        r_int_list = open(os.path.join(path, "mg_init_r.txt"), 'r')
        r_sol_list = open(os.path.join(path, "mg_sol_r.txt"), 'r')
        train_value, train_x = get_MG_matrix(r_int_list, r_sol_list)

    elif (args.benchmark=='Lagos_fine'):
        path ="/home/cc/AutoHPCnet-benchmark/Laghos/fine_grained"
        input_e = np.array(get_Lagos_files(os.path.join(path, "input_e")))
        input_v = np.array(get_Lagos_files(os.path.join(path, "input_v")))
        input_x = np.array(get_Lagos_files(os.path.join(path, "input_x")))
        output_e = np.array(get_Lagos_files(os.path.join(path, "output_e")))
        output_v = np.array(get_Lagos_files(os.path.join(path, "output_v")))
        output_x = np.array(get_Lagos_files(os.path.join(path, "output_x")))
        train_value = np.concatenate((input_e, input_v, input_x), axis=1)
        train_x = np.concatenate((output_e, output_v, output_x), axis=1)

    elif (args.benchmark=='Lagos_coarse'):
        path ="/home/cc/AutoHPCnet-benchmark/Laghos/coarse_grained"
        input_e = np.array(get_Lagos_files(os.path.join(path, "input_e")))
        input_v = np.array(get_Lagos_files(os.path.join(path, "input_v")))
        input_x = np.array(get_Lagos_files(os.path.join(path, "input_x")))
        output_e = np.array(get_Lagos_files(os.path.join(path, "output_e")))
        output_v = np.array(get_Lagos_files(os.path.join(path, "output_v")))
        output_x = np.array(get_Lagos_files(os.path.join(path, "output_x")))
        train_value = np.concatenate((input_e, input_v, input_x), axis=1)
        train_x = np.concatenate((output_e, output_v, output_x), axis=1)

    return train_value,  train_x
