import tensorflow as tf
import numpy as np
import autokeras as ak
import time
from configuration import args
import os
from kerastuner.engine.hyperparameters import Choice

def charbonier_mape_loss(output, gt, epsilon):
    return np.mean(np.abs((output - gt) + epsilon)/(np.abs(gt) + 1e-9))


def Model_search(x_train, y_train, x_test, y_test, x):
    #Initialize the auto regression model.

    # model_dir = os.path.join("auto_model", "position_"+str(x[0]))
    input_node = ak.Input()
    input_length = len(x_train[1])
    output_length = len(y_train[1])
    #input_node = ak.Input()
    hidden_node1 = ak.DenseBlock(num_units=Choice("num_units", [input_length, input_length]))(input_node)
   # hidden_node2 = ak.DenseBlock(num_units=Choice("num_units", [input_length*2, input_length*4]))(hidden_node1)
    hidden_node3 = ak.DenseBlock(num_units=Choice("num_units", [input_length, input_length*2]))(hidden_node1)
    #hidden_node4 = ak.DenseBlock(num_units=Choice("num_units", [int(input_length*0.5), input_length]))(hidden_node3)
    hidden_node5 = ak.DenseBlock(num_units=Choice("num_units", [int(input_length*0.25), int(input_length*0.5)]))(hidden_node3)
    hidden_node6 = ak.DenseBlock(num_units=Choice("num_units", [int((input_length+output_length) * 0.25), int((input_length+output_length) * 0.5)]))(hidden_node5)
    #hidden_node7 = ak.DenseBlock(num_units=Choice("num_units", [output_length*4, output_length*8]))(hidden_node6)
    hidden_node8 = ak.DenseBlock(num_units=Choice("num_units", [output_length * 2, output_length * 4]))(hidden_node5)
    output_node = ak.RegressionHead()(hidden_node8)

    auto_model = ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        overwrite=True,
        # project_name=model_dir,
        tuner="bayesian",
        max_trials=5)

    auto_model.fit(x_train, y_train, epochs=args.numEpoch)

    ml_loss = auto_model.evaluate(x_test, y_test)

    auto_model.export_model()

    return -ml_loss[0]#, timeconsume
