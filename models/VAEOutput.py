from dataclasses import dataclass
import torch

@dataclass
class VAEOutput:
    """
    Extends the dataclass for the VAE's output.
    Attributes:
        zDist: Distribution of the latent variable z.
        zSample: Sampled value for z.
        xReconst: Output reconstructed from z.
        loss: Overall loss of the VAE.
        lossReconst: Component of the loss corresponding to reconstruction.
        lossKL: Component of the loss corresponding to KL divergence.
    """
    zDist: torch.distributions.Distribution
    zSample: torch.Tensor
    xReconst: torch.Tensor
    
    loss: torch.Tensor
    lossReconst: torch.Tensor
    lossKL: torch.Tensor