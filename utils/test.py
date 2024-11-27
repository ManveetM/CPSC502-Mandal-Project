import torch
import tqdm

def test(model : torch.nn.Module, dataLoader : torch.utils.data.DataLoader, curStep : int, latentDim : int, device, writer=None):
    """
    Tests model performance on given data set.
    Args:
        model: Model to test.
        dataLoader: Dataset to test on.
        curStep: Current step of testing.
        latentDim: The dimension of the latent space.
        device: Device to run on.
    """
    # Set model to evaluation mode
    model.eval()
    testLoss = 0
    testReconstLoss = 0
    testKLLoss = 0

    with torch.no_grad():
        for data, target in tqdm(dataLoader, desc='Testing'):
            data = data.to(device)
            # Flatten data
            data = data.view(data.size(0), -1)
            
            # Forward pass
            output = model(data, compute_loss=True)

            testLoss += output.loss.item()
            testReconstLoss += output.lossReconst.item()
            testKLLoss += output.lossKL.item()
    
    testLoss /= len(dataLoader)
    testReconstLoss /= len(dataLoader)
    testKLLoss /= len(dataLoader)
    print(f'----> Test set loss: {testLoss:.4f} (BCE: {testReconstLoss:.4f}, KL: {testKLLoss:.4f})')

    if writer:
        writer.add_scalar('Loss/Test', testLoss, global_step=curStep)
        writer.add_scalar('Loss/Test/BCE', output.lossReconst.item(), global_step=curStep)
        writer.add_scalar('Loss/Test/KLD', output.lossKL.item(), global_step=curStep)
        
        # Log our reconstructions
        writer.add_images('Test/Reconstructions', output.xReconst.view(-1, 1, 28, 28), global_step=curStep)
        writer.add_images('Test/Originals', data.view(-1, 1, 28, 28), global_step=curStep)
        
        # Log the random samples from the latent space
        z = torch.randn(16, latentDim).to(device)
        samples = model.decode(z)
        writer.add_images('Test/Samples', samples.view(-1, 1, 28, 28), global_step=curStep)