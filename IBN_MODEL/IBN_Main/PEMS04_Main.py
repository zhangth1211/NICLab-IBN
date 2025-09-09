"""
    The main function is primarily responsible for training the model on the PEMS04 dataset.
"""
import copy
import os
import numpy as np
# import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader  # DataLoader is responsible for batching and loading input data into the model during training
from matplotlib import pyplot as plt
import random
from Data_metric.Mask_Metric import masked_mae, masked_mape, masked_rmse, masked_mse
from IBN_Model.IBN_Architecture import IBNModel
from Adjacency_Matrix_Calculate.Adjacency_Matrix_Solve import load_adj
import time  # Display the current timestamp

# This is the library needed for cosine annealing learning rate scheduling.
from torch.optim.lr_scheduler import LambdaLR
import math

seed = 7
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)  # Set a seed to 7


def Inverse_normalization(x, max_data, min_data):
    return x * (max_data - min_data) + min_data  # Inverse normalization


data_name = 'PEMS04'
data_file = "../Test_Data" + "/" + data_name + "/data.npz"
raw_data = np.load(data_file, allow_pickle=True)


adj_mx, _ = load_adj("../Test_Data/" + data_name + "/" + "PEMS04_adj.pkl", "doubletransition")
adj_mx = [torch.tensor(i).float() for i in adj_mx]


"""-----------------------------Hyperparameter Define----------------------------"""
batch_size = 16
epoch = 100  # Number of epochs for cosine annealing
max_epoch = 100  # Maximum number of training epochs
IF_mask = 0.25  # missing rates: 0.25, 0.5, and 0.75
lr_rate = 0.003  # learning rate


input_len = 12  # Input length: H
num_id = 307  # Number of nodes: N
out_len = 12  # Output length: L
in_size = 1  # Input feature dimension: C
emb_size = 32  # embedding size C'
grap_size = 16  # graph embedding size D'
layer_num = 2
dropout = 0.1
max_norm = 1.0  # Gradient clipping

# We adopt a cosine annealing learning rate schedule,
# which helps avoid gradient explosion while achieving better training performance.
# The first 10 epochs are warm-up epochs,
# during which the learning rate linearly increases to the maximum learning rate lr_rate.
warmup_epochs = 10


"""-----------------------------Hyperparameter Define Ending----------------------------"""


# Cosine Annealing
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        # Warmup: Linear increase
        return (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing: decays from 1 to 0
        return (1 + math.cos(math.pi * (epoch - warmup_epochs) / (100 - warmup_epochs))) / 2


"""-----------------------------Data Loading------------------------------------------"""
# Train Data Loading
if IF_mask == 0.25:
    train_data = torch.cat([torch.tensor(raw_data["train_x_mask_25"]), torch.tensor(raw_data["train_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.5:
    train_data = torch.cat([torch.tensor(raw_data["train_x_mask_50"]), torch.tensor(raw_data["train_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.75:
    train_data = torch.cat([torch.tensor(raw_data["train_x_mask_75"]), torch.tensor(raw_data["train_y"])], dim=-1).to(torch.float32)
else:
    train_data = torch.cat([torch.tensor(raw_data["train_x_raw"]), torch.tensor(raw_data["train_y"])], dim=-1).to(torch.float32)

train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)


# Valid Data Loading
if IF_mask == 0.25:
    valid_data = torch.cat([torch.tensor(raw_data["vali_x_mask_25"]), torch.tensor(raw_data["vali_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.5:
    valid_data = torch.cat([torch.tensor(raw_data["vali_x_mask_50"]), torch.tensor(raw_data["vali_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.75:
    valid_data = torch.cat([torch.tensor(raw_data["vali_x_mask_75"]), torch.tensor(raw_data["vali_y"])], dim=-1).to(torch.float32)
else:
    valid_data = torch.cat([torch.tensor(raw_data["vali_x_raw"]), torch.tensor(raw_data["vali_y"])], dim=-1).to(torch.float32)

valid_data = DataLoader(valid_data, batch_size=batch_size, shuffle=False)


# Test Data Loading
if IF_mask == 0.25:
    test_data = torch.cat([torch.tensor(raw_data["test_x_mask_25"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.5:
    test_data = torch.cat([torch.tensor(raw_data["test_x_mask_50"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.75:
    test_data = torch.cat([torch.tensor(raw_data["test_x_mask_75"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)
else:
    test_data = torch.cat([torch.tensor(raw_data["test_x_raw"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)

test_data = DataLoader(test_data, batch_size=batch_size, shuffle=False)


max_min = raw_data['max_min']
max_data, min_data = max_min[0], max_min[1]


"""-----------------------------Model Train & Valid------------------------------------------"""
# CPU and GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
my_net = IBNModel(input_len, num_id, out_len, in_size, emb_size, grap_size, layer_num, dropout, adj_mx)
my_net = my_net.to(device)


optimizer = optim.Adam(params=my_net.parameters(), lr=lr_rate)
scheduler = LambdaLR(optimizer, lr_lambda)
min_valid_loss = float("inf")


# Model Training
for i in range(max_epoch):
    num = 0
    loss_out = 0.0
    my_net.train()
    start = time.time()

    for data in train_data:
        my_net.zero_grad()

        train_feature = data[:, :, :, 0:in_size].to(device)
        train_target = data[:, :, :, -1].to(device)
        train_pre = my_net(train_feature)
        loss_data = masked_mae(train_pre, train_target, 0.0)

        loss_data.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(my_net.parameters(), max_norm=max_norm)

        num += 1
        optimizer.step()
        loss_out += loss_data
    loss_out = loss_out / num
    end = time.time()

# Model Validating

    num_va = 0
    loss_valid = 0.0
    my_net.eval()
    with torch.no_grad():
        for data in valid_data:
            valid_x = data[:, :, :, 0:in_size].to(device)
            valid_y = data[:, :, :, -1].to(device)
            valid_pre = my_net(valid_x)
            loss_data = masked_mae(valid_pre, valid_y, 0.0)

            num_va += 1
            loss_valid += loss_data
        loss_valid = loss_valid / num_va  # 计算验证集的lose
        if loss_valid < min_valid_loss:
            best_net = copy.deepcopy(my_net)
            min_valid_loss = loss_valid
            best_epoch = i
            print("The best model occurred in epoch{0}, where the lowest validation loss was{1}".format(best_epoch + 1, min_valid_loss))


# Learn Rate Updating
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {i+1}, Learning Rate: {current_lr:.6f}")
    print('Loss of the {} epoch of the training set: {:02.4f}, Loss of the validation set Loss:{:02.4f}, training time: {:02.4f}:'.format(i+1, loss_out, loss_valid, end - start))


"""-----------------------------Model Saving & Test-----------------------------------------"""
# Model Save
model_save_path = "../IBN_Best_Model_Save/PEMS04_Save/"
torch.save(best_net, model_save_path + "IBN_PEMS04_7_0.25.pt")
with open(model_save_path + "IBN_PEMS04_7_0.25.pt", "rb") as f:
    PEMS04_Best_Model = torch.load(f)
PEMS04_Best_Model.eval()
PEMS04_Best_Model = PEMS04_Best_Model.to(device2)


# Model Test
with torch.no_grad():
    all_pre = 0.0
    all_true = 0.0
    num = 0
    for data in test_data:
        test_feature = data[:, :, :, 0:in_size].to(device2)
        test_target = data[:, :, :, -1].to(device2)
        test_pre = PEMS04_Best_Model(test_feature)
        if num == 0:
            all_pre = test_pre
            all_true = test_target
        else:
            all_pre = torch.cat([all_pre, test_pre], dim=0)
            all_true = torch.cat([all_true, test_target], dim=0)
        num += 1

final_pred = Inverse_normalization(all_pre, max_data, min_data)
final_target = Inverse_normalization(all_true, max_data, min_data)


mae, mape, rmse = masked_mae(final_pred, final_target, 0.0), masked_mape(final_pred, final_target, 0.0)*100, masked_rmse(final_pred, final_target, 0.0)
print('RMSE: {}, MAPE: {}, MAE: {}'.format(rmse, mape, mae))
