import utils.data as data
import utils.train as train
from models.vae import VAE
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

learnRate = 1e-3
weightDecay = 1e-2 # Used to reduce overfitting
num_epochs = 3

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")

model = VAE().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learnRate, weight_decay=weightDecay)
writer = SummaryWriter()
        

writer.close()