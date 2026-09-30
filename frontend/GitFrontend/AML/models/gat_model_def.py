"""
GAT AML Model Architecture Definition — Stage 1
Exact architecture as designed:
  - Linear input projection (13 → 64)
  - 2x GATv2Conv layers (heads=4, out=16 each, concat → 64) with residual + LayerNorm
  - Classification head: [h_src(64) || h_dst(64) || edge(20)] = 148 → 64 → 32 → 1 → sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv
except ImportError:
    raise ImportError(
        "torch_geometric is required. Install with: "
        "pip install torch_geometric"
    )


class GATv2AMLModel(nn.Module):
    """
    Full Stage-1 GAT AML detection model:
      - Encoder: 2-layer GATv2Conv with residual connections and LayerNorm
      - Head: MLP classifier on (h_src || h_dst || edge_features) → AML probability
    """

    NODE_IN_DIM    = 13
    HIDDEN_DIM     = 64
    GAT_HEADS      = 4
    GAT_OUT        = 16
    EDGE_DIM       = 20
    COMBINED_DIM   = 148  # 64 + 64 + 20

    def __init__(self):
        super().__init__()

        self.input_proj = nn.Linear(self.NODE_IN_DIM, self.HIDDEN_DIM)

        self.conv1 = GATv2Conv(
            in_channels  = self.HIDDEN_DIM,
            out_channels = self.GAT_OUT,
            heads        = self.GAT_HEADS,
            concat       = True,
            edge_dim     = self.EDGE_DIM,
            dropout      = 0.1
        )
        self.norm1 = nn.LayerNorm(self.HIDDEN_DIM)

        self.conv2 = GATv2Conv(
            in_channels  = self.HIDDEN_DIM,
            out_channels = self.GAT_OUT,
            heads        = self.GAT_HEADS,
            concat       = True,
            edge_dim     = self.EDGE_DIM,
            dropout      = 0.1
        )
        self.norm2 = nn.LayerNorm(self.HIDDEN_DIM)

        self.classifier = nn.Sequential(
            nn.Linear(self.COMBINED_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def encode(self, x, edge_index, edge_attr):
        h = F.relu(self.input_proj(x))
        h1 = self.conv1(h, edge_index, edge_attr)
        h  = self.norm1(h + h1)
        h2 = self.conv2(h, edge_index, edge_attr)
        h  = self.norm2(h + h2)
        return h

    def classify(self, h_src, h_dst, edge_feat):
        combined = torch.cat([h_src, h_dst, edge_feat], dim=-1)
        logit    = self.classifier(combined)
        return torch.sigmoid(logit).squeeze()

    def forward(self, x, edge_index, edge_attr, src_idx, dst_idx, target_edge_feat):
        h = self.encode(x, edge_index, edge_attr)
        return self.classify(h[src_idx], h[dst_idx], target_edge_feat)
