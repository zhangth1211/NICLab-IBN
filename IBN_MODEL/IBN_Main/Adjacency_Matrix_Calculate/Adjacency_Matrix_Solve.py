"""
The main purpose of this entire Python file is to prepare for the subsequent dataset preprocessing,
by providing utility functions in advance for file loading, adjacency matrix processing, and data information extraction.

Specifically, it includes:
    - Loading data files (e.g., pickle files)
    - Loading and preprocessing the adjacency matrix
    - Loading node2vec embeddings
    - Providing structural information and feature representations for the downstream model
"""


import pickle

import torch
import numpy as np


from IBN_Main.Adjacency_Matrix_Calculate.Adjacency_Matrix_Norm import calculate_scaled_laplacian, calculate_symmetric_normalized_laplacian, calculate_symmetric_message_passing_adj, calculate_transition_matrix


def load_pkl(pickle_file: str) -> object:  # 加载路径文件
    """Load pickle data.
    Args:
        pickle_file (str): file path
    Returns:
        object: loaded objected
    """

    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def dump_pkl(obj: object, file_path: str):

    """Duplicate pickle data.
    Args:
        obj (object): object
        file_path (str): file path
    """

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_adj(file_path: str, adj_type: str):

    """load adjacency matrix.
    Args:
        file_path (str): file path
        adj_type (str): adjacency matrix type
    Returns:
        list of numpy.matrix: list of preproceesed adjacency matrices
        np.ndarray: raw adjacency matrix
    """

    try:
        # METR and PEMS_BAY are traffic speed datasets.
        # METR-LA contains traffic speed data from the Los Angeles region, with sensor nodes having geographical adjacency relationships.
        # PEMS_BAY covers the San Francisco Bay area, featuring more nodes and a more complex structure.
        _, _, adj_mx = load_pkl(file_path)  # Contains three types of information: timestamp, features, and adjacency matrix
    except ValueError:
        # PeMS04
        # Part of the PeMS (PeMSD7) series dataset, it typically contains only the adjacency matrix and time series data. It is a traffic flow dataset.
        adj_mx = load_pkl(file_path)  # PeMS04 contains only the adjacency matrix

    """
    
    Next, process the input data to obtain the corresponding adjacency matrix according to the requirements.
    
    """
    if adj_type == "scalap":
        # Scaled Laplacian matrix, used in models such as GCN, ChebNet, etc.
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        # Symmetric normalized Laplacian matrix — commonly used in graph convolution.
        adj = [calculate_symmetric_normalized_laplacian(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        # Symmetric Message Passing Adjacency — used in models such as GPR-GNN and GPRGCN.
        adj = [calculate_symmetric_message_passing_adj(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        # Transition Matrix: Represents the random walk dynamics, commonly used for message propagation in GNNs.
        adj = [calculate_transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        # Forward and reverse transition matrices — suitable for bidirectional propagation in directed graphs.
        adj = [calculate_transition_matrix(adj_mx).T, calculate_transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        # Identity matrix (self-loop) — used to add self-connections in the graph.
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == "original":
        # Original adjacency matrix — retains the raw graph structure without any preprocessing.
        adj = [adj_mx]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj, adj_mx  # Return the processed matrix and the original adjacency matrix


def load_node2vec_emb(file_path: str) -> torch.Tensor:
    """load node2vec embedding
    Args:
        file_path (str): file path
    Returns:
        torch.Tensor: node2vec embedding
    """

    # spatial embedding

    with open(file_path, mode="r") as f:

        lines = f.readlines()
        temp = lines[0].split(" ")
        num_vertex, dims = int(temp[0]), int(temp[1])
        spatial_embeddings = torch.zeros((num_vertex, dims), dtype=torch.float32)
        for line in lines[1:]:
            temp = line.split(" ")
            index = int(temp[0])
            spatial_embeddings[index] = torch.Tensor([float(ch) for ch in temp[1:]])
    return spatial_embeddings
