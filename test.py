import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch import nn
import pandas as pd

import pandas as pd
data = pd.read_csv("lasso_dataset.csv")
data1 = pd.read_csv("final_data.csv")
final=data1

labels_two = final['A/C'].to_list()
temp=[]
for i in range(len(labels_two)):
    if labels_two[i] == 'A':
        temp.append(1)
    else:
        temp.append(0)
final.drop('A/C', axis=1, inplace=True)
final.drop('Sample_ID', axis=1, inplace=True)
inp = len(data.columns)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

X_train, X_test, y_train, y_test = train_test_split(data, temp, test_size=0.3, shuffle=True)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

train_set = TensorDataset(X_train_tensor, y_train_tensor)
test_set = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=1)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(inp, 100),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        out = self.layer_stack(x)
        return out

model = Network()
model.to(device)

loss_val=[]
running_loss=[]
epochs = 100
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
model.train()
for i in range(epochs):
    for data in tqdm(train_loader):
        batch = tuple(t.to(device) for t in data)
        values, labels = batch
        output = model(values.float())
        #print(f"Output_maxed:{torch.argmax(output, dim=1)}")
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss.append(loss.item())
        #print(f"Loss is :{loss.item()}")
    print(f"Epoch {i}")
    total=0
    for i in range(len(running_loss)): total+=running_loss[i]
    loss_val.append(total/len(running_loss))
    print(f"Loss: {total/len(running_loss)}")

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
    #print(output)
    pred.append(predicted.cpu())
    label.append(labels.cpu())
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = (correct/total)*100
print(f"\nTest Accuracy: {format(accuracy, '.4f')}%\n")

from sklearn.metrics import confusion_matrix
print(confusion_matrix(label, pred))

PATH = "model.pth"
torch.save(model.state_dict(), PATH)





















dat = pd.read_csv("sample.csv")
final=dat
labels_two = final['A/C'].to_list()
temp=[]
for i in range(len(labels_two)):
    if labels_two[i] == 'A':
        temp.append(1)
    else:
        temp.append(0)
final.drop('A/C', axis=1, inplace=True)
#final.drop('Sample_ID', axis=1, inplace=True)
inp = len(final.columns)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

sc = StandardScaler()
X_test = sc.fit_transform(final)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(temp)

test_set = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_set, batch_size=1)

class Newmodel(nn.Module):
    def __init__(self):
        super(Newmodel, self).__init__()
        self.custom_weights = torch.randn(100, 20907)
        self.layer_1 = nn.Linear(inp, 100)
        self.func =  nn.ReLU()
        self.dropout = nn.Dropout(0.8)
        self.layer_2 = nn.Linear(100, 50)
        self.layer_3 = nn.Linear(50, 25)
        self.layer_4 = nn.Linear(25, 8)
        self.layer_5 = nn.Linear(8, 2)
        self.layer_1.weight = nn.Parameter(self.custom_weights, requires_grad=True)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.func(out)
        out = self.dropout(out)
        out = self.layer_2(out)
        out = self.func(out)
        out = self.layer_3(out)
        out = self.func(out)
        out = self.layer_4(out)
        out = self.func(out)
        out = self.layer_5(out)
        return out

Newone = Newmodel()
Newone.to(device)

Newone.load_state_dict(torch.load("model.pth"), strict=False)



model = Network()
model.load_state_dict(torch.load(PATH))
model.to(device)
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
    #print(output)
    pred.append(predicted.cpu())
    label.append(labels.cpu())
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = (correct/total)*100
print(f"\nTest Accuracy: {format(accuracy, '.4f')}%\n")

from sklearn.metrics import confusion_matrix
print(confusion_matrix(label, pred))