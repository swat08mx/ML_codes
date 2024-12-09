import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

tensor_transform = transforms.ToTensor()
dataset = datasets.MNIST(root = "./data",
                         train = True,
                         download = True,
                         transform = tensor_transform)
loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = 32,
                                     shuffle = True)
if torch.cuda.is_available():
  dev='cuda:0'
else:
  dev="cpu"

input_size=784
hidden_enc=[512, 128]
hidden_dec=[512, 784]
class autoenc(nn.Module):
    def __init__(self, hidden_enc, hidden_dec, input_size):
        super().__init__()
        self.layers_enc = nn.ModuleList()
        for size in hidden_enc:
            self.layers_enc.append(nn.Linear(input_size, size))
            input_size = size

        self.layers_dec = nn.ModuleList()
        for size in hidden_dec:
            self.layers_dec.append(nn.Linear(input_size, size))
            input_size = size
        self.layers_dec.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers_enc:
            x = F.relu(layer(x))
        nn.Dropout(0.8)
        counter = 1
        for layer in self.layers_dec:
            x = F.relu(layer(x))
            counter += 1
            if counter == len(self.layers_dec):
                break
        return x

model=autoenc(hidden_enc, hidden_dec, input_size)
model.to(dev)

loss_func = nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
print(f"The model is using the {dev}")
epochs=20
outputs=[]
loss_list=[]
for epoch in range(epochs):
    for (image, _) in tqdm(loader):
        image=image.reshape(-1, 28*28).to(dev)
        reconstructed = model(image)
        loss = loss_func(reconstructed, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.detach().cpu())
    outputs.append((epochs, image.detach().cpu(), reconstructed.detach().cpu()))

plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(loss_list[-100:])

plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(loss_list)

epoch, image, reconstructed = outputs[0]
image= image.detach().cpu()
reconstructed = reconstructed.detach().cpu()
for i, item in enumerate(image):
  item = item.reshape(-1, 28, 28)
  plt.imshow(item[0])
  plt.show()

for i, item in enumerate(reconstructed):
  item = item.reshape(-1, 28, 28)
  plt.imshow(item[0])
  plt.show()