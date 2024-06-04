import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import os
import json

classes = ('uçak', 'araba', 'kuş', 'kedi', 'geyik', 'köpek', 'kurbağa', 'at', 'gemi', 'kamyon')

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
model.load_state_dict(torch.load('deepzeri.pth'))
model.train()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

def train_model(model, optimizer, criterion, trainloader, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Kayıp: {running_loss / len(trainloader)}')
        evaluate_and_update_performance(model, epoch+1)
    
    torch.save(model.state_dict(), 'deepzeri.pth')
    print('Eğitim Tamamlandı')

def evaluate_and_update_performance(model, epoch):
    model.eval()
    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]
    with torch.no_grad():
        for data in trainset:
            images, labels = data
            outputs = model(images.unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            class_total[labels] += 1
            class_correct[labels] += (predicted == labels).sum().item()
    
    class_accuracies = {classes[i]: 100 * class_correct[i] / class_total[i] if class_total[i] != 0 else 0 for i in range(10)}
    update_performance(class_accuracies, epoch)

def update_performance(class_accuracies, epoch):
    if not os.path.exists('performance.json'):
        performance = {class_name: [] for class_name in classes}
        with open('performance.json', 'w') as f:
            json.dump(performance, f)
    with open('performance.json', 'r+') as f:
        data = json.load(f)
        for class_name, accuracy in class_accuracies.items():
            data[class_name].append({'epoch': epoch, 'doğruluğu': accuracy})
        f.seek(0)
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    epochs = 10
    train_model(model, optimizer, criterion, trainloader, epochs)
