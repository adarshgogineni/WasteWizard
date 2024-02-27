import os
import torch
import torchvision
from torch.utils.data import random_split, DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models.resnet import ResNet50_Weights

# Define transformations and dataset
transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
data_dir = 'Garbage classification/Garbage classification'
dataset = ImageFolder(data_dir, transform=transformations)

# Split dataset
random_seed = 42
torch.manual_seed(random_seed)
train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])
batch_size = 32

# Model definition
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)
        return F.cross_entropy(out, labels)
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = sum(batch_accs) / len(batch_accs)
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc}

class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
    
    def forward(self, xb):
        return self.network(xb)

# Accuracy function
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)

# Check device
def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Move data and model to device
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)

def main():
    device = get_default_device()

    # DataLoader should be inside the main function
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    # Evaluation and Training functions
    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        optimizer = opt_func(model.parameters(), lr)
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            result = evaluate(model, val_loader)
            print(f"Epoch {epoch+1}: val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")

    # Instantiate and prepare model
    model = to_device(ResNet(), device)

    # Training model
    num_epochs = 8
    opt_func = torch.optim.Adam
    lr = 5.5e-5
    fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

    # Save the model
    torch.save(model.state_dict(), 'garbage_classification_model.pth')

    # Function to predict image
    def predict_image(image_path, model, device, transformations):
        image = Image.open(image_path)
        image_tensor = transformations(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            # Add the return or process output as needed

if __name__ == '__main__':
    main()
