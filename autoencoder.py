import tensorflow as tf
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import tensorflow.python.framework.dtypes
from tensorflow.keras.models import Model
from tensorflow.keras import layers, regularizers, models, losses
from numpy import vstack

import pandas as pd
import autokeras as ak
import timeit
import time
import pdb

from configuration import args
from bayes_opt import BayesianOptimization
from Visualization_Bayesian import *
from tensorflow.keras.callbacks import ReduceLROnPlateau

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      # layers.Dense(latent_dim*2, activation='sigmoid'),
      layers.Dense(latent_dim,activity_regularizer=regularizers.l1(10e-5)),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(15625, activation='sigmoid'),
      layers.Reshape((125, 125))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


def Embedding_matrix(x_data, encoding_dim):
    # x_data = x_data.reshape((len(x_data), np.prod(x_data.shape[1:])))
    # # This is our input image
    # input = tf.keras.Input(shape=(15625,))
    print(x_data.shape)
    # import pdb; pdb.set_trace()
    latent_dim = encoding_dim
    autoencoder = Autoencoder(latent_dim)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=losses.MeanSquaredError())
    reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.8,
    patience=20,
    min_lr=1e-6,
    verbose=2
    )

    history_reduce_lr = autoencoder.fit(
        x_data, x_data,
        epochs=500,
        batch_size=64,
        shuffle=True,
        verbose=2,
        validation_split=0.20,
        callbacks=[reduce_lr]
    )
    encoded_matrix = autoencoder.encoder(x_data).numpy()
    encoded_matrix.flatten()
    autoencoder.save(os.path.join(args.save_bayesian_path, "autoencoder"))
    return encoded_matrix
