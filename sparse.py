from scipy.sparse import coo_matrix, csr_matrix
import tensorflow as tf
import numpy as np
import timeit
import time

tf.executing_eagerly()
X = np.random.rand(125,125)
print(X)
_, nb_features = X.shape

X_sp = coo_matrix(X)
print(X_sp.row, X_sp.col, X_sp.data)


X_tf_ind = tf.SparseTensor(indices=np.column_stack((X_sp.row, X_sp.col)), values=X_sp.col, dense_shape=X.shape)
X_tf_val = tf.SparseTensor(indices=np.column_stack((X_sp.row, X_sp.col)), values=X_sp.data, dense_shape=X.shape)



embedding_size = 20
V = tf.constant(np.random.random((nb_features, embedding_size)))

print('Truth')
dense_start = time.clock()
X = X_sp.todense()
print(X @ V)
dense_end = time.clock()
dense_timeconsume = dense_end - dense_start
print("\tFunction Dense multiple Time:" + str([round(float(dense_timeconsume), 5)]))

sparse_start = time.clock()
result = tf.nn.embedding_lookup_sparse(V, X_tf_ind, X_tf_val, combiner='sum')
sparse_end = time.clock()
sparse_timeconsume = sparse_end - sparse_start
print("\tFunction embedding_lookup_sparse Time:" + str([round(float(sparse_timeconsume), 5)]))

print('Test')
print(result)
