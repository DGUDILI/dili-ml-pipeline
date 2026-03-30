"""
DGUDILI Cross-Attention module

FP 4 groups (const, pc, maccs, estate) + ChemBERTa CLS
combined via Cross-Attention -> 16-dim feature.

Mode B: Q=ChemBERTa(1 token), K=V=FP(4 tokens)
  -> ChemBERTa selects which FP group to attend to
  -> output: (batch, 1, d_model) -> squeeze -> Linear(d_model, 16)
"""

import torch
import torch.nn as nn


# Fallback dims (overridden at runtime by get_fp_groups)
FP_GROUP_DIMS = {
    "const":  173,
    "pc":       6,
    "maccs":  167,
    "estate":  79,
}


def get_fp_groups(X_df):
    """Feature DataFrame -> 4-group column dict (prefix-based)."""
    cols = X_df.columns.tolist()
    return {
        "const":  [c for c in cols if not c.startswith(("MACCS", "Estate", "PC"))],
        "pc":     [c for c in cols if c.startswith("PC")],
        "maccs":  [c for c in cols if c.startswith("MACCS")],
        "estate": [c for c in cols if c.startswith("Estate")],
    }


class DGUDILIModel(nn.Module):
    """
    FP 4-group Linear Projection + Cross-Attention (Mode B) + 16-dim output.

    Train : forward()          -> logits (batch, 1)   [BCEWithLogitsLoss]
    Infer : extract_features() -> (batch, 16)          [sklearn LR input]
    """

    def __init__(
        self,
        fp_group_dims: dict = None,
        cb_dim: int = 768,
        d_model: int = 64,
        n_heads: int = 4,
        output_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if fp_group_dims is None:
            fp_group_dims = FP_GROUP_DIMS

        self.n_groups = len(fp_group_dims)
        self.d_model  = d_model

        # FP group projections
        self.fp_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
            )
            for dim in fp_group_dims.values()
        ])

        # ChemBERTa CLS projection
        self.cb_projection = nn.Sequential(
            nn.Linear(cb_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Multi-head Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # Mode B: (batch, 1, d_model) -> squeeze -> (batch, d_model) -> 16
        self.head = nn.Linear(d_model, output_dim)

        # Training classifier head
        self.classifier = nn.Linear(output_dim, 1)

    def _cross_attend(self, fp_tensors: list, cb_emb: torch.Tensor) -> torch.Tensor:
        """
        fp_tensors : [(batch, dim_i), ...] x4
        cb_emb     : (batch, 768)
        returns    : (batch, 16)
        """
        # FP: project each group then stack -> (batch, 4, d_model)
        fp_tokens = torch.stack(
            [proj(fp_tensors[i]) for i, proj in enumerate(self.fp_projections)],
            dim=1,
        )

        # CB: project -> (batch, 1, d_model)
        cb_token = self.cb_projection(cb_emb).unsqueeze(1)

        # Mode B: Q=CB, K=V=FP
        attn_out, _ = self.cross_attn(
            query=cb_token,   # (batch, 1, d_model)
            key=fp_tokens,    # (batch, 4, d_model)
            value=fp_tokens,
        )
        # Residual + LayerNorm -> squeeze -> Linear -> (batch, 16)
        attn_out = self.layer_norm(cb_token + self.dropout(attn_out))
        return self.head(attn_out.squeeze(1))

    def forward(self, fp_tensors: list, cb_emb: torch.Tensor) -> torch.Tensor:
        """Train: returns logits (batch, 1)."""
        return self.classifier(self._cross_attend(fp_tensors, cb_emb))

    def extract_features(self, fp_tensors: list, cb_emb: torch.Tensor) -> torch.Tensor:
        """Infer: returns 16-dim features (batch, 16)."""
        with torch.no_grad():
            return self._cross_attend(fp_tensors, cb_emb)
