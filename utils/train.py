import torch
from utils.data import batchSize
from tqdm import tqdm

# Train function
def train(model : torch.nn.Module, dataLoader : torch.utils.data.DataLoader, optimizer, prevUpdates, device, writer=None):
    """
    Train model on given dataset.
    Args:
        model: Model to train.
        dataLoader: Dataset.
        optimizer: The optimizer.
        prevUpdates: Updates from previous run.
        device: The device we want to run on.
    """
    model.train()

    for batchIDX, (data, target) in enumerate(tqdm(dataLoader)):
        update = prevUpdates + batchIDX

        data = data.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = output.loss

        loss.backward()

        if update % 100 == 0:
            # Calculate and print gradient norms
            totalNorm = 0.0
            for p in model.parameters():
                paramNorm = p.grad.data.norm(2) # Compute L2 (Euclidean norm)
                totalNorm += paramNorm.item() ** 2
            totalNorm = totalNorm ** (1.0 / 2.0)

            print(f'Step {update:,} (N samples: {update*batchSize:,}), Loss: {loss.item():.4f} (Recon: {output.lossReconst.item():.4f}), KL: {output.lossKL.item():.4f} Grad: {totalNorm:.4f}')
            
            if writer:
                step = update
                writer.add_scalar('Loss/Train', loss.item(), step)
                writer.add_scalar('Loss/Train/BCE', output.lossReconst.item(), step)
                writer.add_scalar('Loss/Train/KL', output.lossKL.item(), step)
                writer.add_scalar('Grad/Norm/Train', totalNorm, step)
            
        # Gradient clipping to prevent exploding gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
        # Update model parameters
        optimizer.step()

    return prevUpdates + len(dataLoader)