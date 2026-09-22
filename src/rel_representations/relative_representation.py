import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional

class RelativeRepresentation(nn.Module):
    """
    Projects absolute embeddings into a relative coordinate system defined by anchors
    using cosine similarity or Euclidean distance (Moschella et al., ICLR 2023).
    """
    def __init__(
        self,
        anchors: Union[torch.Tensor, np.ndarray],
        similarity: str = "cosine",
        center: bool = True,
        eps: float = 1e-8
    ):
        super().__init__()
        
        if isinstance(anchors, np.ndarray):
            anchors = torch.from_numpy(anchors).float()
            
        assert anchors.ndim == 2, f"Anchors must be 2D [K, d], got shape {anchors.shape}"
        self.similarity = similarity.lower()
        self.center = center
        self.eps = eps

        # Store anchors and anchor mean as persistent buffers (not model parameters)
        self.register_buffer("anchors", anchors)
        if self.center:
            self.register_buffer("anchor_mean", torch.mean(anchors, dim=0, keepdim=True))
        else:
            self.register_buffer("anchor_mean", torch.zeros(1, anchors.shape[1]))

    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Args:
            x: Absolute representations of shape [N, d] or [B, N, d]
        Returns:
            Relative representations of shape [N, K] or [B, N, K]
        """
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x = torch.from_numpy(x).float().to(self.anchors.device)

        # 1. Centering
        if self.center:
            x_centered = x - self.anchor_mean
            anchors_centered = self.anchors - self.anchor_mean
        else:
            x_centered = x
            anchors_centered = self.anchors

        # 2. Similarity Projection
        if self.similarity == "cosine":
            # Normalize vectors to unit length
            x_norm = F.normalize(x_centered, p=2, dim=-1, eps=self.eps)
            anchors_norm = F.normalize(anchors_centered, p=2, dim=-1, eps=self.eps)
            
            # r_x = x_norm @ anchors_norm.T
            relative_rep = torch.matmul(x_norm, anchors_norm.transpose(-1, -2))

        elif self.similarity in ["euclidean", "l2"]:
            # -||x - a||_2 (Negative Euclidean distance as similarity)
            # Using cdist for numerical stability
            dist = torch.cdist(x_centered, anchors_centered, p=2)
            relative_rep = -dist

        elif self.similarity == "inner_product":
            relative_rep = torch.matmul(x_centered, anchors_centered.transpose(-1, -2))

        else:
            raise ValueError(f"Unsupported similarity type: {self.similarity}")

        if is_numpy:
            return relative_rep.cpu().numpy()
        return relative_rep