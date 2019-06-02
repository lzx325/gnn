import tensorflow as tf
import numpy as np
import Library
import GNN as GNN
import Net_Clique as n
import tensorflow as tf
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.preprocessing import StandardScaler

data_path = "./Data/Clique"
################ training set#################

inp, arcnode, nodegraph, nodein , labels = Library.set_load_clique(data_path, "train")
###########validation set#####################

inp_val, arcnode_val, nodegraph_val, nodein_val, labels_val = Library.set_load_clique(data_path, "validation")
###########test set#####################

inp_test, arcnode_test, nodegraph_test, nodein_test, labels_test = Library.set_load_clique(data_path, "test")

# set threshold, learning rate and state dimension
threshold = 0.001
learning_rate = 0.0001
state_dim = 5

# set input and output dim, the maximum number of iterations, the number of epochs and the optimizer
tf.reset_default_graph()

input_dim = len(inp[0][0])
output_dim = 2
max_it = 50
num_epoch = 5000
optimizer = tf.train.AdamOptimizer

# initialize state and output network
net = n.Net(input_dim, state_dim, output_dim)

# initialize GNN
param = "st_d" + str(state_dim) + "_th" + str(threshold) + "_lr" + str(learning_rate)
print(param)
g = GNN.GNN(net, max_it=max_it, input_dim=input_dim, output_dim=output_dim, state_dim=state_dim, optimizer=optimizer, learning_rate=learning_rate, threshold=threshold)
count = 0

# train the model and validate every 30 epochs
for j in range(0, num_epoch):
    g.Train(inp[0], arcnode[0], labels, count, nodegraph[0])

    if count % 30 == 0:
        print(g.Validate(inp_val[0], arcnode_val[0], labels_val, count, nodegraph_val[0]))

    count = count + 1

# evaluate on the test set
print(g.Evaluate(inp_test[0], arcnode_test[0], labels_test, nodegraph_test[0]))
