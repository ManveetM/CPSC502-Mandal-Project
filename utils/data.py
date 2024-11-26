import torch
from torchvision import datasets
from torchvision.transforms import v2

batchSize = 128
transform = v2.Compose([
    v2.ToImage(),                           # Convert to Tensor
    v2.ToDtype(torch.float32, scale=True),  # Scale pixel values to [0, 1]
    v2.Lambda(lambda x: x.view(-1))         # Flatten image
])

# Load the training data
trainData = datasets.FashionMNIST(
    './data/', download=True, train=True, transform=transform
    )

# Load the test data
testData = datasets.FashionMNIST(
    './data/', download=True, train=False, transform=transform
)

# Create dataloaders
trainLoader = torch.utils.data.DataLoader(
    trainData, batch_size=batchSize, shuffle=True
)

testLoader = torch.utils.data.DataLoader(
    testData, batch_size=batchSize, shuffle=False
)