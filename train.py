import torch
import torchvision
from torch.utils.data import DataLoader, Subset

transforms = torchvision.transforms
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])

def train(model, model_file_path):
  import torch.optim as optim

  nn = torch.nn

  trainset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
  )
  subset_indices = torch.randperm(len(trainset))[:int(0.4 * len(trainset))]
  trainset = Subset(trainset, subset_indices)
  trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      if i % 100 == 99: # print every 100 mini-batches
        print('[%d, %5d] loss: %.3f' % (
          epoch + 1,
          i + 1,
          running_loss / 100
        ))
        running_loss = 0.0

    print('Finished Training')
    torch.save(model.state_dict(), model_file_path)
    print("Model saved successfully.")

def test(model):
  import matplotlib.pyplot as plt
  import numpy as np

  testset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
  )
  subset_indices_test = torch.randperm(len(testset))[:int(0.4 * len(testset))]
  testset = Subset(testset, subset_indices_test)
  testloader = DataLoader(testset, batch_size=32, shuffle=False)

  correct = 0
  total = 0
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print(f"""Accuracy of the network on the {total} test images:
    {100 * correct / total}
  """)

  def imshow(img):
      img = img / 2 + 0.5 # unnormalize
      npimg = img.numpy()
      plt.imshow(np.transpose(npimg, (1, 2, 0)))
      plt.show()

  # Get some random test images
  dataiter = iter(testloader)
  images, labels = next(dataiter) # Get the next batch of data

  print(images)

  # print and visualize the first 8 images
  print('GroundTruth: ', ' '.join('%5s' % labels[j].item() for j in range(8)))

  outputs = model(images)
  _, predicted = torch.max(outputs, 1)

  print('Predicted: ', ' '.join('%5s' % predicted[j].item() for j in range(8)))

  imshow(torchvision.utils.make_grid(images[:8]))

