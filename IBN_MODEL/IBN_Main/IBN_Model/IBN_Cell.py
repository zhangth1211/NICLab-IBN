import torch
from torch import nn, optim
import torch.nn.functional as F


class IBNCell(nn.Module):
    def __init__(self, num_id, in_size, emb_size, grap_size, dropout):
        """
        Args:
            num_id : Number of variables in the input data, corresponding to N.
            in_size : Input feature dimension, corresponding to C.
            emb_size : Embedding dimension, corresponding to C'.
            grap_size : Graph dimension size.
            dropout : Dropout rate for regularization.
        """
        super(IBNCell, self).__init__()
        self.emb_size = emb_size
        self.num_id = num_id
        self.emb = nn.Conv1d(in_channels=in_size, out_channels=emb_size, kernel_size=1)
        self.emb2 = nn.Linear(num_id, num_id)
        self.att = InterpositionAttention(emb_size, emb_size, num_id, grap_size, dropout)
        self.uai_test = UAI_Module(emb_size, emb_size)
        self.linear1 = nn.Conv1d(in_channels=emb_size, out_channels=emb_size, kernel_size=1, bias=False)
        self.linear2 = nn.Conv1d(in_channels=emb_size, out_channels=emb_size, kernel_size=1, bias=True)
        self.layernorm = nn.LayerNorm([emb_size, num_id])
        self.dropout = nn.Dropout(dropout)
        self.GL = nn.Parameter(torch.FloatTensor(num_id, grap_size))
        nn.init.kaiming_uniform_(self.GL)
        self.GL_linear = nn.Linear(grap_size, emb_size, bias=False)
        self.GL_linear2 = nn.Linear(emb_size * 2, emb_size * 2, bias=False)

    def forward(self, x, ct, graph_data):
        """
        Args:
            x (Tensor): Input data with shape [B, C, H, N].
                        However, for each cell, only data at a single time step is input: [B, C, N].
            ct (Tensor): Corresponding cell state with shape [N, 1, C'].
            graph_data (Tensor): Bidirectional adjacency matrix with shape [2, N, N].

        """
        x = self.att(self.emb2(F.leaky_relu(self.emb(x))).transpose(-2, -1))  # [B,C',N]

        # UAI module
        uai_mu, uai_sigma = self.uai_test(x)
        x_uai = (uai_mu / (1 + uai_sigma)).transpose(-2, -1)

        # GGCN
        # Predefined graph
        graph_data1 = graph_data[0].to(x.device) + torch.eye(self.num_id).to(x.device)
        graph_data1 = IBNCell.calculate_laplacian_with_self_loop(graph_data1)
        graph_data2 = graph_data[1].to(x.device) + torch.eye(self.num_id).to(x.device)
        graph_data2 = IBNCell.calculate_laplacian_with_self_loop(graph_data2)

        # Adaptive graph using RBF (Gaussian kernel) based on Euclidean distance
        B, _, _ = x_uai.shape
        x_feat = x_uai.transpose(-2, -1)  # [B, N, C']
        dist = torch.cdist(x_feat, x_feat, p=2)  # [B, N, N]
        sigma = self.emb_size ** 0.5
        adj = torch.exp(-dist ** 2 / (2 * sigma ** 2))  # [B, N, N]
        adj = adj + torch.eye(self.num_id).to(x.device)
        graph_learn = F.softmax(adj, dim=-1)

        # IBN cell
        x_new = self.layernorm(self.dropout(self.linear1(x_uai)) @ graph_learn + self.linear1(
            self.dropout(self.linear1(x_uai)) @ graph_data1) @ graph_data2)
        ft = F.gelu(self.layernorm(self.dropout(self.linear2(x_uai)) @ graph_learn + self.linear2(
            self.dropout(self.linear2(x_uai)) @ graph_data1) @ graph_data2))
        rt = F.gelu(self.layernorm(self.dropout(self.linear2(x_uai)) @ graph_learn + self.linear2(
            self.dropout(self.linear2(x_uai)) @ graph_data1) @ graph_data2))
        ct = ft * ct + x_new - ft * x_new
        ht = rt * F.elu(ct) + x_uai - rt * x_uai
        return ht, ct

    @staticmethod
    def calculate_laplacian_with_self_loop(matrix):
        row_sum = matrix.sum(1)  # [N,1]
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)  # [N，1] --> [N,N]
        normalized_laplacian = (
            matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
        )
        return normalized_laplacian


class InterpositionAttention(nn.Module):
    def __init__(self, in_c, out_c, num_id, grap_size, dropout):
        """
            Inductive Attention
            :param in_c: embeding size  corresponds to feature dimension C in input shape [N, H, C]
            :param out_c: embeding output size corresponds to feature dimension C' in output shape [N, H, C']
            :param num_id: Number of nodes — denoted as N.
            :param grap_size:  Graph dimension size
            :param dropout: dropout
        """

        super(InterpositionAttention, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_id = num_id
        self.drop = dropout
        self.dropout = nn.Dropout(dropout)

        self.W = nn.Parameter(torch.FloatTensor(size=(in_c, out_c)))
        nn.init.xavier_uniform_(self.W)
        self.a = nn.Parameter(torch.FloatTensor(size=(2 * out_c, 1)))
        nn.init.xavier_uniform_(self.a)

        self.leakyrelu = nn.LeakyReLU()
        self.GL = nn.Parameter(torch.FloatTensor(num_id, grap_size))
        nn.init.kaiming_uniform_(self.GL)  # [N, d]

        self.GL2 = nn.Parameter(torch.FloatTensor(grap_size, num_id))
        nn.init.kaiming_uniform_(self.GL2)  # [d, N]

    def forward(self, inp):
        """
        inp: input_fea [B, N, C]
        """

        adj = F.softmax(F.relu(self.GL @ self.GL.transpose(-2, -1)),
                        dim=-1)
        B, N = inp.size(0), inp.size(1)
        adj = adj + torch.eye(N, dtype=adj.dtype, device=adj.device)  # Add identity matrix to emphasize self-correlation.

        h = torch.matmul(inp, self.W)
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_c)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -1e12 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = self.dropout(F.softmax(attention, dim=2))
        h_prime = torch.matmul(attention, self.dropout(h))
        h_prime = F.relu(h_prime)
        return h_prime.transpose(-2, -1)


class UAI_Module(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_samples=10, dropout_p=0.1):
        """
        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            num_samples (int): Number of Monte Carlo sampling iterations.
            dropout_p (float): dropout_p (float): Dropout probability for Monte Carlo Dropout
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.num_samples = num_samples

    def forward(self, x_ia):
        """
            Args:
                x_ia: [B, C', N]
        """
        x_ia = x_ia.transpose(-2, -1)
        preds = []
        for _ in range(self.num_samples):
            out = self.dropout(self.linear(x_ia))  # [B, N, C'] -> [B, N, hidden_dim]
            preds.append(out.unsqueeze(-1))  # [B, N, hidden_dim, 1]
        stacked = torch.cat(preds, dim=-1)  # [B, N, hidden_dim, num_samples]
        mu = stacked.mean(dim=-1)
        sigma = stacked.std(dim=-1)
        return mu, sigma
