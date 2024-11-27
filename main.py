import utils.data as data
import utils.train as train
import utils.test as test
from models.vae import VAE
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

learnRate = 1e-3
weightDecay = 1e-2 # Used to reduce overfitting
numEpochs = 16

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")

model = VAE().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learnRate, weight_decay=weightDecay)
writer = SummaryWriter()

# Run train/test loop
prevUpdate = 0
latentDim = 64       
for epoch in range(numEpochs):
    print(f'Epch {epoch+1}/{numEpochs}')
    prevUpdate = train.train(model, data.trainLoader, optimizer, prevUpdate, device, writer=writer)
    test.test(model, data.testLoader, prevUpdate, latentDim, device, writer=writer)

writer.close()