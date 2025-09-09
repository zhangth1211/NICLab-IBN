"""
    This section of code is primarily responsible for preprocessing the PEMS_BAY dataset
    and setting up the corresponding adjacency matrix, facilitating subsequent training.
"""

import numpy as np
import copy
import random


import h5py


import os


import pickle
import csv
import pandas as pd

seed = 42
random.seed(seed)
np.random.seed(seed)   # Set a random seed

folder_path = "../../Test_Data/PEMS_BAY/"
data_file_path = folder_path + "pems-bay.h5"

"""--------------- Data Loading -------------------"""
with h5py.File(data_file_path, 'r') as f:
    # 'r' stands for read-only mode
    # The file contains only one key: ['speed']
    # ['speed'] contains four data components: ['axis0', 'axis1', 'block0_items', 'block0_values]
    group = f['speed']
    raw_data_axis0 = group['axis0'][()]  # Contains the variable node IDs, type: numpy.ndarray, shape: [325,]; node IDs are stored as int64
    raw_data_axis1 = group['axis1'][()]  # Time data stored as nanosecond Unix timestamps, type: numpy.ndarray, shape: [52116,]
    raw_data_block0_items = group['block0_items'][()]  # Contains variable node IDs with shape [325,], identical to data_axis0; either can be used, but only one is needed
    raw_data_block0_values = group['block0_values'][()]  # The data has shape [52116, 325], where each row represents a timestamp and each column represents a variable.

raw_time_data = raw_data_axis1
feature_data = raw_data_block0_values
# Since the time is in Unix timestamp format, convert it to human-readable (normal) time
normal_time_data = (raw_time_data.astype('datetime64[ns]')).astype('str')


history_seq_len = 12  # Input sequence length H
future_seq_len = 12  # Prediction sequence length L
# 60% training set, 20% validation set, 20% test set
train_ratio = 0.6
valid_ratio = 0.2
mask_ratio = 0.25  # The proportion of data to be masked, hidden, or ignored


# Data split
l, n = feature_data.shape
print("Total number of variables: {0}".format(n))
num_samples = feature_data.shape[0] - history_seq_len - future_seq_len + 1


# Calculate the number of samples in the training, validation, and test sets
train_num_short = round(num_samples * train_ratio)
valid_num_short = round(num_samples * valid_ratio)
test_num_short = num_samples - train_num_short - valid_num_short

mask_samples_1 = round(n * mask_ratio)

# Display the number of samples in the training, validation, and test sets, as well as the number of missing variables
print("number of training samples:{0}".format(train_num_short))
print("number of validation samples:{0}".format(valid_num_short))
print("number of test samples:{0}".format(test_num_short))
print("number of masx samples:{0}".format(mask_samples_1))


# Normalize the feature data
def normalize(x, max_data, min_data):
    return (x - min_data) / (max_data - min_data)    # Min-Max normalization


data_max, data_min = feature_data.max(), feature_data.min()
max_min = [data_max, data_min]
max_min = np.array(max_min)


"""----------------0.25 Mask rate-------------------"""
mask_id_1 = random.sample(range(n), mask_samples_1)
mask_id_1 = sorted(mask_id_1)

print("ID of mask samples in 0.25 mask rate:{0}".format(mask_id_1))

target_data = feature_data[:, mask_id_1]
missing_data = copy.deepcopy(feature_data)
missing_data[:, mask_id_1] = 0


# normalize
data_new = normalize(feature_data, data_max, data_min)
missing_new = normalize(missing_data, data_max, data_min)
target_new = normalize(target_data, data_max, data_min)


# Partition dataset
def feature_target(original_data, input_len, output_len):
    """
    Args:
        original_data : Input dataset with shape [L, N].
            It has only one feature (e.g., univariate time series per node).
        input_len : Length of the input sequence (number of time steps fed into the model).
        output_len : Length of the output/prediction sequence (number of future time steps to predict).

    Returns:
        tuple: A tuple containing:
            - fin_input : Reshaped input data with shape [NUM, input_len, N].
            - fin_output : Corresponding output (target) data with shape [NUM, output_len, N].
    """
    fin_input = []
    fin_output = []
    data_len = original_data.shape[0]
    for i in range(data_len - input_len - output_len + 1):
        lin_in_seq = original_data[i:i + input_len, :]
        lin_out_seq = original_data[i + input_len:i + input_len + output_len, :]
        fin_input.append(lin_in_seq)
        fin_output.append(lin_out_seq)
    fin_input = np.array(fin_input).transpose((0, 2, 1))
    fin_output = np.array(fin_output).transpose((0, 2, 1))
    return fin_input, fin_output


fin_input_x, fin_output_y = feature_target(data_new, history_seq_len, future_seq_len)


"""--------------data without mask------------"""
# Train_data
train_x_raw = fin_input_x[0:train_num_short, :, :]
train_y = fin_output_y[0:train_num_short, :, :]
train_y = train_y.transpose(0, 2, 1)
train_y = np.expand_dims(train_y, axis=-1)

# Valid_data
vali_x_raw = fin_input_x[train_num_short:train_num_short+valid_num_short, :, :]
vali_y = fin_output_y[train_num_short:train_num_short+valid_num_short, :, :]
vali_y = vali_y.transpose(0, 2, 1)
vali_y = np.expand_dims(vali_y, axis=-1)

# Test_data
test_x_raw = fin_input_x[train_num_short+valid_num_short:, :, :]
test_y = fin_output_y[train_num_short+valid_num_short:, :, :]
test_y = test_y.transpose(0, 2, 1)
test_y = np.expand_dims(test_y, axis=-1)

mask_id_1 = np.array(mask_id_1)


"""------------------data with 0.25 mask------------------"""
mask_input_x, _ = feature_target(missing_new, history_seq_len, future_seq_len)

# train
train_x_mask_1 = mask_input_x[0:train_num_short, :, :]
train_x_mask_1 = train_x_mask_1.transpose(0, 2, 1)
train_x_mask_1 = np.expand_dims(train_x_mask_1, axis=-1)

# valid
vali_x_mask_1 = mask_input_x[train_num_short:train_num_short+valid_num_short, :, :]
vali_x_mask_1 = vali_x_mask_1.transpose(0, 2, 1)
vali_x_mask_1 = np.expand_dims(vali_x_mask_1, axis=-1)

# test
test_x_mask_1 = mask_input_x[train_num_short+valid_num_short:, :, :]
test_x_mask_1 = test_x_mask_1.transpose(0, 2, 1)
test_x_mask_1 = np.expand_dims(test_x_mask_1, axis=-1)


"""------------------0.5 Mask rate------------------"""
mask_ratio_2 = 0.5
mask_samples_2 = round(n * mask_ratio_2)

mask_id_2 = random.sample(range(n), mask_samples_2)
mask_id_2 = sorted(mask_id_2)

print("ID of mask samples in 0.5 mask rate:{0}".format(mask_id_2))

missing_data_2 = copy.deepcopy(feature_data)
missing_data_2[:, mask_id_2] = 0

missing_new_2 = normalize(missing_data_2, data_max, data_min)
mask_input_x_2, _ = feature_target(missing_new_2, history_seq_len, future_seq_len)


# train
train_x_mask_2 = mask_input_x_2[0:train_num_short, :, :]
train_x_mask_2 = train_x_mask_2.transpose(0, 2, 1)
train_x_mask_2 = np.expand_dims(train_x_mask_2, axis=-1)

# valid
vali_x_mask_2 = mask_input_x_2[train_num_short:train_num_short+valid_num_short, :, :]
vali_x_mask_2 = vali_x_mask_2.transpose(0, 2, 1)
vali_x_mask_2 = np.expand_dims(vali_x_mask_2, axis=-1)

# test
test_x_mask_2 = mask_input_x_2[train_num_short+valid_num_short:, :, :]
test_x_mask_2 = test_x_mask_2.transpose(0, 2, 1)
test_x_mask_2 = np.expand_dims(test_x_mask_2, axis=1)

mask_id_2 = np.array(mask_id_2)


"""------------------0.75 Mask rate------------------"""

mask_ratio_3 = 0.75
mask_samples_3 = round(n * mask_ratio_3)

mask_id_3 = random.sample(range(n), mask_samples_3)
mask_id_3 = sorted(mask_id_3)

print("ID of mask samples in 0.75 mask rate:{0}".format(mask_id_3))

missing_data_3 = copy.deepcopy(feature_data)
missing_data_3[:, mask_id_3] = 0

missing_new_3 = normalize(missing_data_3, data_max, data_min)
mask_input_x_3, _ = feature_target(missing_new_3, history_seq_len, future_seq_len)


# train
train_x_mask_3 = mask_input_x_3[0:train_num_short, :, :]
train_x_mask_3 = train_x_mask_3.transpose(0, 2, 1)
train_x_mask_3 = np.expand_dims(train_x_mask_3, axis=-1)

# valid
vali_x_mask_3 = mask_input_x_3[train_num_short:train_num_short+valid_num_short, :, :]
vali_x_mask_3 = vali_x_mask_3.transpose(0, 2, 1)
vali_x_mask_3 = np.expand_dims(vali_x_mask_3, axis=-1)

# test
test_x_mask_3 = mask_input_x_3[train_num_short+valid_num_short:, :, :]
test_x_mask_3 = test_x_mask_3.transpose(0, 2, 1)
test_x_mask_3 = np.expand_dims(test_x_mask_3, axis=-1)


mask_id_3 = np.array(mask_id_3)


"""------------------data_save------------------"""
"""
    Due to server performance limitations and to reduce GPU memory pressure, 
    it's not feasible to store all missing rate data simultaneously.
    Instead, only data for a specific missing rate is saved, 
    facilitating efficient storage and model training.
"""
# data_save
np.savez(folder_path + "data.npz",
         train_x_raw=train_x_raw,
         train_x_mask_25=train_x_mask_1,
         # train_x_mask_50=train_x_mask_2,
         # train_x_mask_75=train_x_mask_3,
         train_y=train_y,


         vali_x_raw=vali_x_raw,
         vali_x_mask_25=vali_x_mask_1,
         # vali_x_mask_50=vali_x_mask_2,
         # vali_x_mask_75=vali_x_mask_3,
         vali_y=vali_y,


         test_x_raw=test_x_raw,
         test_x_mask_25=test_x_mask_1,
         # test_x_mask_50=test_x_mask_2,
         # test_x_mask_75=test_x_mask_3,
         test_y=test_y,


         max_min=max_min,
         mask_id_25=mask_id_1,
         # mask_id_50=mask_id_2,
         # mask_id_75=mask_id_3,
         )


"""------------------Checking & Constructing a Predefined Adjacency Matrix------------------"""


def Adjacency_Matrix_Generate(csv_path: str, node_name_list: list, node_num: int, normalized_k=0.1):
    """
    Args:
        csv_path (str): Path to the CSV file containing edge information (e.g., from, to, distance/cost).
        node_name_list (list or np.ndarray): List of node names/IDs in the desired order.
            Used to reorder nodes when constructing the adjacency matrix, as the original order may be inconsistent.
        node_num (int): Total number of nodes in the graph.
        normalized_k (float): Threshold value for sparsification. After normalizing the distance matrix,
            entries smaller than `normalized_k` are set to 0 to enforce sparsity.

    Returns:
        adj_mx (np.ndarray): The constructed adjacency matrix with shape [node_num, node_num].
    """
    distance_csv = pd.read_csv(csv_path)
    data_adj = np.zeros([node_num, node_num], dtype=float)
    data_adj[:] = np.inf

    # Generate the adjacency matrix based on the structure of distance_csv (from, to, cost)
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(node_name_list):
        sensor_id_to_ind[int(sensor_id)] = i

    # Fills cells in the matrix with distances
    row_num = distance_csv.shape[0]
    for i in range(row_num):
        if int(distance_csv.iloc[i, 0]) not in sensor_id_to_ind or int(distance_csv.iloc[i, 1]) not in sensor_id_to_ind:
            continue
        data_adj[sensor_id_to_ind[int(distance_csv.iloc[i, 0])], sensor_id_to_ind[int(distance_csv.iloc[i, 1])]] = distance_csv.iloc[i, 2]

    # Adjacency Matrix Normalize
    distances = data_adj[~np.isinf(data_adj)].flatten()
    std = distances.std()
    data_adj = np.exp(-np.square(data_adj / std))

    # For the PEMS_BAY dataset, the adjacency matrix does not need diagonalized
    data_adj[data_adj < normalized_k] = 0  # Sparsify the adjacency matrix

    return data_adj


test_adj = Adjacency_Matrix_Generate(folder_path+"distances_bay_2017.csv", raw_data_axis0, n)
with open(folder_path + "PEMS_BAY_adj.pkl", 'wb') as f:
    pickle.dump(test_adj, f)
print("The generated matrix has dimensions {0}".format(test_adj.shape))
print(test_adj)


