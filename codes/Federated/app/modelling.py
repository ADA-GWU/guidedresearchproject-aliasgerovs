import torch.nn.functional as F
from torch import nn
from torch import optim
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch

class net2nn(nn.Module):
    def __init__(self):
        super(net2nn, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_model(self, train_loader, criterion, optimizer):
        self.train() 
        train_loss = 0.0 
        correct = 0 
        for data, target in train_loader: 
            output = self(data) 
            loss = criterion(output, target)
            optimizer.zero_grad() 
            loss.backward()  
            optimizer.step()
            train_loss += loss.item() 
            prediction = output.argmax(dim=1, keepdim=True) 
            correct += prediction.eq(target.view_as(prediction)).sum().item()

        return train_loss / len(train_loader), correct / len(train_loader.dataset) 

    def validate_model(self, test_loader, criterion):
        self.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self(data)
                test_loss += criterion(output, target).item()
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(target.view_as(prediction)).sum().item()

        test_loss /= len(test_loader)
        correct /= len(test_loader.dataset)
        return test_loss, correct