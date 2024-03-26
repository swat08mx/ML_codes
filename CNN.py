import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import pandas as pd

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

data = pd.read_csv("/content/drive/MyDrive/final_data.csv")
# data=data.iloc[:, :5000]
temp = []
for i in range(len(data['A/C'])):
    if data['A/C'][i] == 'A':
        temp.append(1)
    else:
        temp.append(0)
data.drop('A/C', axis=1, inplace=True)
data.drop('Sample_ID', axis=1, inplace=True)
# temp = pd.DataFrame(temp, columns=['labels'])
final=data

X_train, X_test, y_train, y_test = train_test_split(final, temp, test_size=0.3, shuffle=True)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

train_set = TensorDataset(X_train_tensor, y_train_tensor)
test_set = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_set, batch_size=16)
test_loader = DataLoader(test_set, batch_size=1)


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(4709 * 4709, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Assuming binary classification

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        print(x.shape)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        print(x.shape)
        torch.flatten(x)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
model = CNN()

loss_val=[]
epochs = 20
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
model.train()
for i in range(epochs):
    for data in tqdm(train_loader):
        batch = tuple(t.to(device) for t in data)
        values, labels = batch
        values = values.view(-1, 18836)
        output = model(values.float())
        print(f"Output_maxed:{torch.argmax(output, dim=1)}")
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_val.append(loss.item())
        print(f"Loss is :{loss.item()}")
    print(f"Epoch {i}")

model.eval()
correct = 0
total = 0
pred=[]
label=[]
for data in tqdm(test_loader):
    batch = tuple(t.to(device) for t in data)
    values, labels = batch
    output = model(values.float())
    predicted = torch.argmax(output, dim=1)
    print(output)
    pred.append(predicted)
    label.append(labels)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = (correct/total)*100
print(f"\nTest Accuracy: {format(accuracy, '.4f')}%\n")