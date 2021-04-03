import tensorflow as tf
import numpy as np
import math
import os
import matplotlib.pyplot as pyplot
from matplotlib import gridspec
import tensorflow.python.framework.dtypes
from tensorflow.keras.models import Model
from tensorflow.keras import layers, regularizers, models, losses
from numpy import vstack
from tensorflow.keras.callbacks import ReduceLROnPlateau

def charbonier_mape_loss(output, gt, epsilon):
    return np.mean(np.abs((output - gt) + epsilon)/(np.abs(gt) + 1e-9))


class MLP_model(Model):
    def  __init__(self, latent_dim, input_dim):
        super()

def Manual_model(x_train, y_train, x_test, y_test, x):
    input_length = len(x_train[1])
    output_length = len(y_train[1])
    hidden_node1 = input_length+output_length
    hidden_node2 = input_length*2+output_length
    hidden_node3 = input_length+output_length
    hidden_node4 = input_length*0.5+output_length
    hidden_node5 = input_length*0.25+output_length
    model = tf.keras.Sequential([
        layers.Dense(input_length, activation='sigmoid'),
        layers.Dense(hidden_node1, activation='sigmoid'),
        layers.Dense(hidden_node2, activation='sigmoid'),
        layers.Dense(hidden_node3, activation='sigmoid'),
        layers.Dense(hidden_node4, activation='sigmoid'),
        layers.Dense(hidden_node5, activation='sigmoid'),
        layers.Dense(output_length, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1), loss=losses.MeanSquaredError())
    reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.8,
    patience=50,
    min_lr=1e-6,
    verbose=2
    )
    # import pdb; pdb.set_trace()
    history = model.fit(
        x_train, y_train,
        epochs=500,
        batch_size=64,
        shuffle=True,
        verbose=2,
        validation_split=0.20,
        callbacks=[reduce_lr]
    )
    # plot learning curves
    # pyplot.title('Learning Curves')
    # pyplot.xlabel('Epoch')
    # pyplot.ylabel('Cross Entropy')
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='val')
    # pyplot.legend()
    # pyplot.show()

    tf.saved_model.save(model, "./Manual_model")
    ml_loss = model.evaluate(x_test, y_test)
    return -ml_loss
