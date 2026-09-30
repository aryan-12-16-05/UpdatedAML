"""
GAT AML Model Architecture Definition — Stage 1
Exact architecture matching the checkpoint state_dict.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv
except ImportError:
    raise ImportError("torch_geometric is required.")

class GATv2AMLModel(nn.Module):
    """
    Full Stage-1 GAT AML detection model matched EXACTLY to the checkpoint keys:
      - node_proj, gat1, norm1, gat2, norm2, classifier
    """

    NODE_IN_DIM    = 13
    HIDDEN_DIM     = 64
    GAT_HEADS      = 4
    GAT_OUT        = 16
    EDGE_DIM       = 20
    COMBINED_DIM   = 148  # 64 + 64 + 20

    def __init__(self):
        super().__init__()

        self.node_proj = nn.Linear(self.NODE_IN_DIM, self.HIDDEN_DIM)

        self.gat1 = GATv2Conv(
            in_channels  = self.HIDDEN_DIM,
            out_channels = self.GAT_OUT,
            heads        = self.GAT_HEADS,
            concat       = True,
            edge_dim     = self.EDGE_DIM,
            dropout      = 0.2
        )
        self.norm1 = nn.LayerNorm(self.HIDDEN_DIM)

        self.gat2 = GATv2Conv(
            in_channels  = self.HIDDEN_DIM,
            out_channels = self.GAT_OUT,
            heads        = self.GAT_HEADS,
            concat       = True,
            edge_dim     = self.EDGE_DIM,
            dropout      = 0.2
        )
        self.norm2 = nn.LayerNorm(self.HIDDEN_DIM)

        self.classifier = nn.Sequential(
            nn.Linear(self.COMBINED_DIM, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def encode(self, x, edge_index, edge_attr):
        h = F.relu(self.node_proj(x))
        h1 = self.gat1(h, edge_index, edge_attr)
        h  = self.norm1(h + h1)
        h2 = self.gat2(h, edge_index, edge_attr)
        h  = self.norm2(h + h2)
        return h

    def classify(self, h_src, h_dst, edge_feat):
        combined = torch.cat([h_src, h_dst, edge_feat], dim=-1)
        logit    = self.classifier(combined)
        return torch.sigmoid(logit).squeeze(-1)

    def forward(self, x, edge_index, edge_attr, src_idx, dst_idx, target_edge_feat):
        h = self.encode(x, edge_index, edge_attr)
        return self.classify(h[src_idx], h[dst_idx], target_edge_feat)
