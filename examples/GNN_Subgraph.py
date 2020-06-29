import tensorflow as tf
import numpy as np
import gnn.gnn_utils as gnn_utils
import gnn.GNN as GNN
import examples.Net_Subgraph as n
from scipy.sparse import coo_matrix

##### GPU & stuff config
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
data_path = "./Data"
#data_path = "./Clique"
# 15 is #nodes in each connected component, 7 is #nodes in subgraph pattern, 200 is number of components
# in this example, even nodes are in different connected components, they are still regarded as in one graph
set_name = "sub_15_7_200"
############# training set ################

#inp, arcnode, nodegraph, nodein, labels = Library.set_load_subgraph(data_path, "train")

# inp contains list of batches like [p_id, c_id, feature_p (feature_dims), feature_c (feature_dims)]], len(inp): n_batches
# arcnode contains list of batches of arcnode matrix (sparse), each of size: (n_nodes,n_edges), len(arcnode):n_batches
# nodegraph contains list of batches of nodegraph matrix len(nodegraph): n_batches, nodegraph[i].shape: (n_nodes,n_graphs)
# nodein contains list of one-dim arrays. Each contains the number of nodes each graph has in this batch. len(nodein): n_batches, nodein[i].shape: (n_graphs,)
# labels (n_nodes,n_classes) one-hot encoded target
# labs (n_nodes,feature_dims) stores the node features

inp, arcnode, nodegraph, nodein, labels, _ = gnn_utils.set_load_general(data_path, "train", set_name=set_name)
############ test set ####################

#inp_test, arcnode_test, nodegraph_test, nodein_test, labels_test = Library.set_load_subgraph(data_path, "test")
inp_test, arcnode_test, nodegraph_test, nodein_test, labels_test, _ = gnn_utils.set_load_general(data_path, "test", set_name=set_name)

############ validation set #############

#inp_val, arcnode_val, nodegraph_val, nodein_val, labels_val = Library.set_load_subgraph(data_path, "valid")
inp_val, arcnode_val, nodegraph_val, nodein_val, labels_val, _ = gnn_utils.set_load_general(data_path, "validation", set_name=set_name)

# set input and output dim, the maximum number of iterations, the number of epochs and the optimizer

threshold = 0.001
learning_rate = 0.001
state_dim = 10

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
g = GNN.GNN(net, max_it=max_it, input_dim=input_dim, output_dim=output_dim, state_dim=state_dim, optimizer=optimizer,
            learning_rate=learning_rate, threshold=threshold, param=param, config=config)
count = 0

# train the model and validate every 30 epochs
for j in range(0, num_epoch):
    g.Train(inp[0], arcnode[0], labels, count, nodegraph[0])

    if count % 30 == 0:
        print("Epoch ", count)
        print("Training: ", g.Validate(inp[0], arcnode[0], labels, count, nodegraph[0]))
        print("Validation: ", g.Validate(inp_val[0], arcnode_val[0], labels_val, count, nodegraph_val[0]))

    count = count + 1

# evaluate on the test set
print(g.Evaluate(inp_test[0], arcnode_test[0], labels_test, nodegraph_test[0]))
