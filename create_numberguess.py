# https://www.datatechnotes.com/2024/04/mnist-image-classification-with-pytorch.html
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os


# Define transforms to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the MNIST training data, then randomly select 40% of it for loading.
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
subset_indices = torch.randperm(len(trainset))[:int(0.4 * len(trainset))]
trainset = Subset(trainset, subset_indices)

# Download and load the MNIST test data, then randomly select 40% of it for loading.
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
subset_indices_test = torch.randperm(len(testset))[:int(0.4 * len(testset))]
testset = Subset(testset, subset_indices_test)

# Define the data loaders
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Save the trained model
model_name = './mnist_cnn.pth'

# Train the model if it doesn't exist
if not os.path.exists(model_name):
    # Training the model
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')
    torch.save(model.state_dict(), model_name)
    print("Model saved successfully.")

# Load the model
model = CNN()
model.load_state_dict(torch.load(model_name))
model.eval()

# Testing the model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        # Determine the predicted class for each image
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the {total} test images: {100 * correct / total}")

# Predicting a new test image with the loaded model
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Get some random test images
dataiter = iter(testloader)
images, labels = next(dataiter)  # Get the next batch of data

print(images)

# print and visualize the first 8 images
print('GroundTruth: ', ' '.join('%5s' % labels[j].item() for j in range(8)))

# Predict the class for images
outputs = model(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % predicted[j].item() for j in range(8)))

# Show images
imshow(torchvision.utils.make_grid(images[:8]))
