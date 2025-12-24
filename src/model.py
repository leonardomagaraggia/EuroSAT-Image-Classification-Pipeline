import torch
import torch.nn as nn
import torch.nn.functional as F

# Definizione della CNN

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Blocchi convoluzionali
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # EuroSAT 64x64 px -> dopo 3 pool 8x8
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout per evitare overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(-1, 128 * 8 * 8)  # flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Funzione per ottenere il device (MPS su Mac ARM, CUDA su GPU NVIDIA, CPU altrimenti)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Test modello

if __name__ == "__main__":
    device = get_device()
    model = SimpleCNN(num_classes=10).to(device)
    print(model)
    
    # Test forward con batch fittizio
    x = torch.randn(2, 3, 64, 64).to(device)  # batch di 2 immagini
    y = model(x)
    print("Output shape:", y.shape)  # deve essere [2, 10]
