"""
    This section of code is primarily responsible for preprocessing the PEMS04 dataset
    and setting up the corresponding adjacency matrix, facilitating subsequent training.
"""

import numpy as np
import copy
import random

# The following libraries are used to save the adjacency matrix as a .pkl file
import pickle
import csv
import pandas as pd

seed = 42
random.seed(seed)
np.random.seed(seed)  # Set a random seed

data_name = "PEMS04"
folder_path = "../../Test_Data/" + data_name + "/"
data_file_path = folder_path + 'PEMS04.npz'


history_seq_len = 12  # Input sequence length H
future_seq_len = 12  # Prediction sequence length L
# 60% training set, 20% validation set, 20% test set
train_ratio = 0.6
valid_ratio = 0.2
target_channel = 0  # target channel(s): the index (dimension) of the feature(s) to predict, here at dimension 0


mask_ratio = 0.25  # The proportion of data to be masked, hidden, or ignored
raw_data = np.load(data_file_path)
data = np.load(data_file_path)["data"]
data = data[..., target_channel]


# Data split
l, n = data.shape  # Contains the sequence length and the number of variables
print("Total number of variables: {0}".format(n))
num_samples = data.shape[0] - history_seq_len - future_seq_len + 1


# Calculate the number of samples in the training, validation, and test sets
train_num_short = round(num_samples * train_ratio)
valid_num_short = round(num_samples * valid_ratio)
test_num_short = num_samples - train_num_short - valid_num_short


mask_samples_1 = round(n * mask_ratio)

print("number of training samples:{0}".format(train_num_short))
print("number of validation samples:{0}".format(valid_num_short))
print("number of test samples:{0}".format(test_num_short))
print("number of masx samples:{0}".format(mask_samples_1))


# Normalize the feature data
def normalize(x, max_data, min_data):
    return (x - min_data) / (max_data - min_data)   # Min-Max normalization


data_max, data_min = data.max(), data.min()
max_min = [data_max, data_min]
max_min = np.array(max_min)


"""----------------0.25 Mask rate-------------------"""
mask_id_1 = random.sample(range(n), mask_samples_1)
mask_id_1 = sorted(mask_id_1)   # Sort the variables by their indices

print("ID of mask samples in 0.25 mask rate:{0}".format(mask_id_1))

target_data = data[:, mask_id_1]
feature_data = copy.deepcopy(data)
feature_data[:, mask_id_1] = 0

# normalize
data_new = normalize(data, data_max, data_min)
feature_new = normalize(feature_data, data_max, data_min)
target_new = normalize(target_data, data_max, data_min)


# Partition dataset
def feature_target(original_data, input_len, output_len):
    """
    Args:
        original_data : The raw input data.
        input_len : Length of the input sequence, in terms of time steps.
        output_len : Length of the output (prediction) sequence, in terms of time steps.
    """
    fin_feature = []
    fin_target_in = []
    data_len = original_data.shape[0]
    for i in range(data_len-input_len - output_len + 1):
        lin_fea_seq = original_data[i:i+input_len, :]
        lin_tar_seq = original_data[i+input_len:i+input_len + output_len, :]
        fin_feature.append(lin_fea_seq)
        fin_target_in.append(lin_tar_seq)
    fin_feature = np.array(fin_feature).transpose((0, 2, 1))
    fin_target_in = np.array(fin_target_in).transpose((0, 2, 1))  # [NUM,N,output_len]
    return fin_feature, fin_target_in


raw_feature, fin_target = feature_target(data_new, history_seq_len, future_seq_len)  # 开始将原始数据进行切割


"""--------------data without mask------------"""
# Train_data
train_x_raw = raw_feature[0:train_num_short, :, :]
train_y = fin_target[0:train_num_short, :, :]
train_y = train_y.transpose(0, 2, 1)
train_y = np.expand_dims(train_y, axis=-1)

# valid_data
vali_x_raw = raw_feature[train_num_short:train_num_short+valid_num_short, :, :]
vali_y = fin_target[train_num_short:train_num_short+valid_num_short, :, :]
vali_y = vali_y.transpose(0, 2, 1)
vali_y = np.expand_dims(vali_y, axis=-1)

# test_data
test_x_raw = raw_feature[train_num_short+valid_num_short:, :, :]
test_y = fin_target[train_num_short+valid_num_short:, :, :]
test_y = test_y.transpose(0, 2, 1)
test_y = np.expand_dims(test_y, axis=-1)

mask_id_1 = np.array(mask_id_1)

"""------------------data with mask------------------"""
mask_feature, _ = feature_target(feature_new, history_seq_len, future_seq_len)

# train
train_x_mask_1 = mask_feature[0:train_num_short, :, :]
train_x_mask_1 = train_x_mask_1.transpose(0, 2, 1)
train_x_mask_1 = np.expand_dims(train_x_mask_1, axis=-1)

# valid
vali_x_mask_1 = mask_feature[train_num_short:train_num_short+valid_num_short, :, :]
vali_x_mask_1 = vali_x_mask_1.transpose(0, 2, 1)
vali_x_mask_1 = np.expand_dims(vali_x_mask_1, axis=-1)

# test
test_x_mask_1 = mask_feature[train_num_short+valid_num_short:, :, :]
test_x_mask_1 = test_x_mask_1.transpose(0, 2, 1)
test_x_mask_1 = np.expand_dims(test_x_mask_1, axis=-1)


"""------------------0.5 Mask rate------------------"""

mask_ratio_2 = 0.5
mask_samples_2 = round(n * mask_ratio_2)

mask_id_2 = random.sample(range(n), mask_samples_2)
mask_id_2 = sorted(mask_id_2)
print("ID of mask samples in 0.5 mask rate:{0}".format(mask_id_2))

feature_data_2 = copy.deepcopy(data)
feature_data_2[:, mask_id_2] = 0

feature_new_2 = normalize(feature_data_2, data_max, data_min)
mask_feature_2, _ = feature_target(feature_new_2, history_seq_len, future_seq_len)

# train
train_x_mask_2 = mask_feature_2[0:train_num_short, :, :]
train_x_mask_2 = train_x_mask_2.transpose(0, 2, 1)
train_x_mask_2 = np.expand_dims(train_x_mask_2, axis=-1)

# valid
vali_x_mask_2 = mask_feature_2[train_num_short:train_num_short+valid_num_short, :, :]
vali_x_mask_2 = vali_x_mask_2.transpose(0, 2, 1)
vali_x_mask_2 = np.expand_dims(vali_x_mask_2, axis=-1)

# test
test_x_mask_2 = mask_feature_2[train_num_short+valid_num_short:, :, :]
test_x_mask_2 = test_x_mask_2.transpose(0, 2, 1)
test_x_mask_2 = np.expand_dims(test_x_mask_2, axis=-1)

mask_id_2 = np.array(mask_id_2)


"""------------------0.75 Mask rate------------------"""

mask_ratio_3 = 0.75
mask_samples_3 = round(n * mask_ratio_3)
mask_id_3 = random.sample(range(n), mask_samples_3)
mask_id_3 = sorted(mask_id_3)
print("ID of mask samples in 0.75 mask rate:{0}".format(mask_id_3))

feature_data_3 = copy.deepcopy(data)
feature_data_3[:, mask_id_3] = 0

feature_new_3 = normalize(feature_data_3, data_max, data_min)
mask_feature_3, _ = feature_target(feature_new_3, history_seq_len, future_seq_len)

# train
train_x_mask_3 = mask_feature_3[0:train_num_short, :, :]
train_x_mask_3 = train_x_mask_3.transpose(0, 2, 1)
train_x_mask_3 = np.expand_dims(train_x_mask_3, axis=-1)

# valid
vali_x_mask_3 = mask_feature_3[train_num_short:train_num_short+valid_num_short, :, :]
vali_x_mask_3 = vali_x_mask_3.transpose(0, 2, 1)
vali_x_mask_3 = np.expand_dims(vali_x_mask_3, axis=-1)

# test
test_x_mask_3 = mask_feature_3[train_num_short+valid_num_short:, :, :]
test_x_mask_3 = test_x_mask_3.transpose(0, 2, 1)
test_x_mask_3 = np.expand_dims(test_x_mask_3, axis=-1)

mask_id_3 = np.array(mask_id_3)


"""------------------data_save------------------"""
# data_save
np.savez(folder_path + "data.npz",
         train_x_raw=train_x_raw,
         train_x_mask_25=train_x_mask_1,
         train_x_mask_50=train_x_mask_2,
         train_x_mask_75=train_x_mask_3,
         train_y=train_y,

         vali_x_raw=vali_x_raw,
         vali_x_mask_25=vali_x_mask_1,
         vali_x_mask_50=vali_x_mask_2,
         vali_x_mask_75=vali_x_mask_3,
         vali_y=vali_y,

         test_x_raw=test_x_raw,
         test_x_mask_25=test_x_mask_1,
         test_x_mask_50=test_x_mask_2,
         test_x_mask_75=test_x_mask_3,
         test_y=test_y,

         max_min=max_min,
         mask_id_25=mask_id_1,
         mask_id_50=mask_id_2,
         mask_id_75=mask_id_3,
         )


"""------------------Constructing a Predefined Adjacency Matrix------------------"""


def Adjacency_Matrix_Generate(csv_path: str, save_adj_path: str, node_num: int, normalized_k=0.1):
    """
    Args:
        csv_path (str): Path to the input CSV file containing edge data (e.g., from, to, distance).
        save_adj_path (str): Path to save the generated adjacency matrix as a .pkl file.
        node_num (int): Number of nodes in the graph.
        normalized_k (float): Threshold value for sparsification. After normalizing the distance matrix,
            entries smaller than this value are set to 0.

    Returns:
        data_adj (np.ndarray): The constructed adjacency matrix with shape [N, N].
    """
    distance_csv = pd.read_csv(csv_path)

    data_adj = np.zeros([node_num, node_num], dtype=float)
    data_adj[:] = np.inf
    # Generate the adjacency matrix from the distance_csv file with columns (from, to, cost).
    # Each row contains: source node (first), target node (second), and distance/cost (third).
    row_num = distance_csv.shape[0]
    for i in range(row_num):
        data_adj[distance_csv.iloc[i, 0], distance_csv.iloc[i, 1]] = distance_csv.iloc[i, 2]
        data_adj[distance_csv.iloc[i, 1], distance_csv.iloc[i, 0]] = distance_csv.iloc[i, 2]

    # Adjacency Matrix Normalize
    data_distance = data_adj[~np.isinf(data_adj)].flatten()
    std = data_distance.std()
    data_adj = np.exp(-np.square(data_adj / std))
    data_adj = np.maximum.reduce([data_adj, data_adj.T])
    data_adj[data_adj < normalized_k] = 0


    with open(save_adj_path, 'wb') as f:
        pickle.dump(data_adj, f)

    return data_adj


csv_name = "distance"
distance_path = folder_path + csv_name + ".csv"
predefined_pkl_path = folder_path + data_name + "_adj.pkl"
PEMS04_adj = Adjacency_Matrix_Generate(distance_path, predefined_pkl_path, n)
print("The adjacency matrix has dimensions {0}".format(PEMS04_adj.shape))
print(PEMS04_adj)

