# type: ignore
import torch
import torch.nn as nn
from typing import Optional


class BosonMPS(nn.Module):
    """
    Translation-invariant MPS: single block contracted with input, then chained L times with trace.
    GTN-only model (not compatible with NTN due to shared parameters).
    """

    def __init__(
        self,
        L: int,
        bond_dim: int,
        phys_dim: int,
        output_dim: int,
        output_site: Optional[int] = None,
        init_strength: float = 0.001,
    ):
        super().__init__()
        
        self.L = L
        self.bond_dim = bond_dim
        self.phys_dim = phys_dim
        self.output_dim = output_dim
        self.output_site = output_site if output_site is not None else 0

        self.block = nn.Parameter(
            torch.randn(bond_dim, phys_dim, bond_dim) * init_strength
        )
        
        self.block_out = nn.Parameter(
            torch.randn(bond_dim, phys_dim, bond_dim, output_dim) * init_strength
        )

        self.input_dims = ["x0"]
        self.output_dims = ["out"]

    def forward(self, x):
        """
        x: (batch, phys_dim) - single input tensor
        
        1. Contract input with block: A[b,l,r] * x[b,p] -> M[b,l,r] (batch, bond, bond)
        2. Chain M L times: M^L with trace
        3. Output site uses block_out for output dimension
        """
        batch_size = x.shape[0]
        
        M = torch.einsum('bp,lpr->blr', x, self.block)
        M_out = torch.einsum('bp,lpro->blro', x, self.block_out)
        
        result = None
        for i in range(self.L):
            if i == self.output_site:
                site_mat = M_out
            else:
                site_mat = M
            
            if result is None:
                result = site_mat
            else:
                if i == self.output_site:
                    result = torch.einsum('blr,brio->blio', result, site_mat)
                elif result.dim() == 4:
                    result = torch.einsum('blio,bri->blro', result, site_mat)
                else:
                    result = torch.einsum('blr,bri->bli', result, site_mat)
        
        if result.dim() == 4:
            output = torch.einsum('bllo->bo', result)
        else:
            output = torch.einsum('bll->b', result).unsqueeze(-1)
        
        return output