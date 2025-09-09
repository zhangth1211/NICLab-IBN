import torch
from torch import nn, optim
import torch.nn.functional as F

from IBN_Main.IBN_Model.IBN_Cell import IBNCell


class IBNModel(nn.Module):
    def __init__(self, input_len, num_id, out_len, in_size, emb_size, grap_size, layer_num, dropout, adj_mx):
        """
        Args:
            input_len (int): Input sequence length, denoted as H.
            num_id (int): Number of input variables (nodes), denoted as N.
            out_len (int): Output sequence length, denoted as L.
            in_size (int): Input feature dimension, denoted as C.
            emb_size (int): Embedding dimension, denoted as C'.
            grap_size : Graph-related dimension in dynamic graph structure D.
            layer_num (int, optional): Number of layers. Default: 2.
            dropout (float, optional): Dropout probability for regularization.
            adj_mx (Tensor, optional): Prior adjacency matrix with shape [N, N].
        """
        super(IBNModel, self).__init__()
        self.input_len = input_len
        self.out_len = out_len
        self.num_id = num_id
        self.layer_num = layer_num
        self.emb_size = emb_size
        self.graph_data = adj_mx

        # encoder
        self.IBN_first = IBNCell(num_id, in_size, emb_size, grap_size, dropout)
        self.IBN_back = IBNCell(num_id, in_size, emb_size, grap_size, dropout)
        self.IBN_other = IBNCell(num_id, 2 * emb_size, 2 * emb_size, grap_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.lay_norm = nn.LayerNorm([input_len, num_id])

        # decoder
        self.decoder = nn.Conv2d(in_channels=layer_num, out_channels=out_len, kernel_size=(1, 2 * emb_size))  # [B,L,N,1]
        self.output = nn.Conv2d(in_channels=out_len, out_channels=out_len, kernel_size=1)  # [B,L,N,1]

    def forward(self, x_input):
        """

        :param x_input: [B,H,N,C]: B is batch size. N is the number of variables. H is the history length. C is the number of feature.
        :return: [B,L,N]: B is batch size. N is the number of variables. L is the future length
        """
        x = x_input.transpose(-3, -1).transpose(-2, -1)
        B, C, L, N = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        # encoder
        final_result = None
        current_layer_output = x
        for z in range(self.layer_num):
            result = None
            result_back = None
            ct = torch.zeros(B, self.emb_size, N).to(x.device)
            ct_back = torch.zeros(B, self.emb_size, N).to(x.device)
            if z == 0:
                # Bi-RU
                for j in range(self.input_len):
                    ht, ct = self.IBN_first(x[:, :, j, :], ct, self.graph_data)
                    ht_back, ct_back = self.IBN_back(x[:, :, self.input_len - 1 - j, :], ct_back, self.graph_data)
                    ht = ht.unsqueeze(-2)
                    ht_back = ht_back.unsqueeze(-2)
                    result = ht if result is None else torch.cat([result, ht], dim=-2)
                    result_back = ht_back if result_back is None else torch.cat([result_back, ht_back], dim=-2)
                result_back_reverse = result_back.flip(dims=[-2])
                current_layer_output = torch.cat([result, result_back_reverse], dim=1)  # [B, 2C', H, N]
            x = current_layer_output
            last_ht = current_layer_output[:, :, -1, :]  # [B, 2C', N]
            if z == 0:
                final_result = last_ht.transpose(-2, -1).unsqueeze(1)
            else:
                final_result = torch.cat([final_result, last_ht.transpose(-2, -1).unsqueeze(1)], dim=1)

        # decoder
        x = self.dropout(self.decoder(final_result))  # [B, L, N, 1]
        x = self.output(x)  # [B, L, N, 1]
        return x.squeeze(-1)  # [B, L, N]
