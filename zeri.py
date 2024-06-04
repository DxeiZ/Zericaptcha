import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from flask import Flask, jsonify, request, render_template
import random
from random import shuffle
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import os
import json
import threading

app = Flask(__name__)

classes = ('uçak', 'araba', 'kuş', 'kedi', 'geyik', 'köpek', 'kurbağa', 'at', 'gemi', 'kamyon')

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

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
model.eval()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=1.0)

def get_image_base64(image):
    npimg = image.cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0)) / 2 + 0.5
    npimg = (npimg * 255).astype(np.uint8)
    img = Image.fromarray(npimg)
    
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=100)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_captcha', methods=['GET'])
def generate_captcha():
    start_time = time.time()
    grid_images = []
    grid_labels = []
    target_label = random.randint(0, 9)
    target_class = classes[target_label]

    target_indices = [i for i in range(len(testset)) if testset[i][1] == target_label]
    selected_target_indices = random.sample(target_indices, 3)

    for index in selected_target_indices:
        image, label = testset[index]
        grid_images.append((image, label))

    other_indices = [i for i in range(len(testset)) if testset[i][1] != target_label]
    selected_other_indices = random.sample(other_indices, 6)

    for index in selected_other_indices:
        image, label = testset[index]
        grid_images.append((image, label))

    shuffle(grid_images)

    with ThreadPoolExecutor() as executor:
        grid_images_base64 = list(executor.map(lambda x: get_image_base64(x[0]), grid_images))

    grid_labels = [label for _, label in grid_images]
    end_time = time.time()
    print(f"Görüntü oluşturma süresi: {end_time - start_time} saniye")

    return jsonify({
        'images': grid_images_base64,
        'labels': grid_labels,
        'target_class': target_class
    })

def train_model(model, optimizer, criterion, data, target):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def update_performance(class_accuracies):
    if not os.path.exists('performance.json'):
        performance = {class_name: [] for class_name in classes}
        with open('performance.json', 'w') as f:
            json.dump(performance, f)
    with open('performance.json', 'r+') as f:
        data = json.load(f)
        for class_name, accuracy in class_accuracies.items():
            data[class_name].append(accuracy)
        f.seek(0)
        json.dump(data, f, indent=4)

def evaluate_and_update_model():
    model.eval()
    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]
    with torch.no_grad():
        for data in testset:
            images, labels = data
            outputs = model(images.unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            class_total[labels] += 1
            class_correct[labels] += (predicted == labels).sum().item()
    
    class_accuracies = {classes[i]: 100 * class_correct[i] / class_total[i] if class_total[i] != 0 else 0 for i in range(10)}
    update_performance(class_accuracies)

@app.route('/validate_captcha', methods=['POST'])
def validate_captcha():
    data = request.json
    selected_indices = data['selected_indices']
    grid_labels = list(map(int, data['grid_labels']))
    target_class = data['target_class']
    
    selected_classes = [classes[grid_labels[i]] for i in selected_indices]
    target_label = classes.index(target_class)
    correct = (selected_classes.count(target_class) == 3)

    if correct:
        inputs = torch.stack([testset[i][0] for i in selected_indices])
        targets = torch.tensor([grid_labels[i] for i in selected_indices], dtype=torch.long)
        criterion = torch.nn.CrossEntropyLoss()
        
        threading.Thread(target=train_model, args=(model, optimizer, criterion, inputs, targets)).start()
        threading.Thread(target=evaluate_and_update_model).start()
        
        return jsonify({'result': "true"})
    else:
        return jsonify({'result': "false"})

if __name__ == '__main__':
    app.run(debug=True)
