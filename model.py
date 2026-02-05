# =========================
# file: model_fedlag.py
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Fast GCN:
    - Default forward returns ONLY logits (no x_dis).
    - If return_embed=True, returns (logits, embedding) where embedding is the hidden representation.
    """
    def __init__(self, feat_dim, hid_dim, out_dim, dropout, num_layers=2):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.conv1 = GCNConv(feat_dim, hid_dim)

        if self.num_layers == 2:
            self.conv2 = GCNConv(hid_dim, hid_dim)
            self.fc = nn.Linear(hid_dim, out_dim)
        else:
            self.fc = nn.Linear(hid_dim, out_dim)

        self.dropout = dropout

    def forward(self, data, return_embed: bool = False):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 2:
            x_hid = self.conv2(x, edge_index)  # [N, hid_dim]
        else:
            x_hid = x

        embed = F.relu(x_hid)
        logits = self.fc(embed)

        if return_embed:
            return logits, embed
        return logits

    @torch.no_grad()
    def encode(self, data):
        """Return node embeddings only (no logits), useful for analysis."""
        _, embed = self.forward(data, return_embed=True)
        return embed


class FedTAD_ConGenerator(nn.Module):
    def __init__(self, noise_dim, feat_dim, out_dim, dropout):
        super(FedTAD_ConGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.emb_layer = nn.Embedding(out_dim, out_dim)

        hid_layers = []
        dims = [noise_dim + out_dim, 64, 128, 256]
        for i in range(len(dims) - 1):
            d_in = dims[i]
            d_out = dims[i + 1]
            hid_layers.append(nn.Linear(d_in, d_out))
            hid_layers.append(nn.Tanh())
            hid_layers.append(nn.Dropout(p=dropout, inplace=False))

        self.hid_layers = nn.Sequential(*hid_layers)
        self.nodes_layer = nn.Linear(256, feat_dim)

    def forward(self, z, c):
        z_c = torch.cat((self.emb_layer.forward(c), z), dim=-1)
        hid = self.hid_layers(z_c)
        node_logits = self.nodes_layer(hid)
        return node_logits


# (Optional) torch.empty patch (some environments need it)
_empty = torch.empty
def _empty_no_none(*size, dtype=None, device=None, **kwargs):
    kw = dict(kwargs)
    if dtype is not None:
        kw["dtype"] = dtype
    if device is not None:
        kw["device"] = device
    return _empty(*size, **kw)

torch.empty = _empty_no_none
