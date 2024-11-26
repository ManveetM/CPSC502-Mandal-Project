import numpy as np
import torch
import torch.nn as nn
from VAEOutput import VAEOutput

class VAE(nn.Module):
    def __init__(self, inputDim : int=784, hiddenDim : int=400, latentDim : int=32):
        """
        Class for our Variational Autoencoder (VAE).
        Args:
            inputDim: The dimension of the input data (default is for 28x28 pixel images).
            hiddenDim: The dimension of the hidden layer for encoder.
            latentDim: The dimension of the latent space.
        """
        super(VAE, self).__init__()

        # Define encoder model
        self.encoder = nn.Sequential(
            # First we gradually decrease the input dimensionality of our data
            # We use the Swish (SiLU) activation function for better peformance
            nn.Linear(inputDim, hiddenDim),
            nn.SiLU, 
            nn.Linear(hiddenDim, hiddenDim // 2),
            nn.SiLU,
            nn.Linear(hiddenDim // 2, hiddenDim // 4),
            nn.SiLU,
            nn.Linear(hiddenDim // 4, hiddenDim // 8),
            nn.SiLU,
            nn.Linear(hiddenDim, 2 * latentDim) # 2 for mean and variance (log var)
        )
        self.softPlus = nn.Softplus

        # Define decoder model
        self.decoder = nn.Sequential(
            nn.Linear(latentDim, hiddenDim // 8),
            nn.SiLU,
            nn.Linear(hiddenDim // 8, hiddenDim // 4),
            nn.SiLU,
            nn.Linear(hiddenDim // 4, hiddenDim // 2),
            nn.SiLU,
            nn.Linear(hiddenDim // 2, hiddenDim),
            nn.SiLU,
            nn.Linear(hiddenDim, inputDim),
            nn.Sigmoid
        )

    def encode(self, x : torch.Tensor, epsilon : float) -> torch.distributions.MultivariateNormal:
        """
        Used to encode the data into the latent space.
        Args:
            x: The input data.
            epsilon: Tiny value to help avoid instability by enforcing lower bound.
        Return:
            A multivariate normal distribution (parameterized by mean and variance) 
            which represents the latent space.
        """ 

        x = self.encoder(x)
        mean, logvar = torch.chunk(x, 2, dim=-1) # Split along the last dimension since we have 2 * latentDim.
        scale = self.softPlus(logvar) + epsilon # Transform the log variance into a non-negative standard deviation.
        scaleTril = torch.diag_embed(scale) # Represent the covariance matrix (assuming its diagonal). 

        return torch.distributions.MultivariateNormal(mean, scale_tril=scaleTril)
        
    def reparam(self, distribution : torch.distributions.MultivariateNormal) -> torch.Tensor:
        """
        Used to reparameterize the encoded data in the latent space so we can sample from it. This needs
        to be done to make it computationally tractable to compute the gradients otherwise we have a random
        term that breaks back propagation. 
        Args:
            distribution: Normal distribution of the latent space.
        Returns:
            Sampled data point from the latent space.
        """
        
        return distribution.rsample()
    
    def decode(self, z : torch.Tensor) -> torch.Tensor:
        """
        Used to decode data from the latent space back into the input space.
        Args:
            z: Data point from the latent space.
        Returns:
            Reconstructed data point in the input space.
        """

        return self.decoder(z)

    def forward(self, x : torch.Tensor, calcLoss : bool=True) -> VAEOutput:
        """
        Forward pass of our VAE.
        Args:
            x: Input data.
            calcLoss: Compute loss or not.
        Returns:
            A class representing the VAEs output.
        """

        distr = self.encode(x)
        z = self.reparam(distr)
        reconstX = self.decode(z)
        
        # Don't compute loss vars
        if calcLoss == False:
            return VAEOutput(
                zDist = distr,
                zSample = z,
                xReconst = reconstX,
                loss = None,
                lossReconst = None,
                lossKL = None
            )
        
        # Compute loss vars
        # First we are calculating loss of black/white pixel values for each individual image then getting the average.
        lossReconst = nn.functional.binary_cross_entropy(reconstX, x, reduction='none').sum(-1).mean()
        # Next we create a standard normal distribution with similar shape to our learned latent distribution.
        stdNormal = torch.distributions.MultivariateNormal(torch.zeros_like(z, device=z.device), 
                                                           scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1))
        # Compute the KL divergence between a normal distribution and the learned latent distribution.
        lossKL = torch.distributions.kl.kl_divergence(distr, stdNormal).mean()

        # Total loss
        loss = lossReconst + lossKL

        return VAEOutput(
                zDist = distr,
                zSample = z,
                xReconst = reconstX,
                loss = loss,
                lossReconst = lossReconst,
                lossKL = lossKL
            )