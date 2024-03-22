import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch import nn
import pandas as pd

import torch
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
data1 = pd.read_csv("final_data.csv")
# data1=data1.iloc[:, :5000]
print(data1)
temp = []
for i in range(len(data1['A/C'])):
    if data1['A/C'][i] == 'A':
        temp.append(1)
    else:
        temp.append(0)
data1.drop('A/C', axis=1, inplace=True)
data1.drop('Sample_ID', axis=1, inplace=True)
# temp = pd.DataFrame(temp, columns=['labels'])
final=data1

# new=[]
# neu=[]
# for i in range(len(data1['A/C'])):
#     if data1['A/C'][i] == 'A':
#         new.append(i)
#     else:
#         neu.append(i)
# #print(f"The A is {len(new)} and C is {len(neu)}")
# temp = neu[:200]
# df = data1.iloc[new]
# df1 = data1.iloc[temp]
# final = pd.concat([df,df1], ignore_index=False)
# labels_two = final['A/C'].to_list()
# temp=[]
# for i in range(len(labels_two)):
#     if labels_two[i] == 'A':
#         temp.append(1)
#     else:
#         temp.append(0)
# #temp = pd.DataFrame(temp, columns=['labels'])
# final.drop('A/C', axis=1, inplace=True)
# final.drop('Sample_ID', axis=1, inplace=True)

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

train_loader = DataLoader(train_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=1)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(18836, 15000),
            nn.ReLU(),
            nn.Linear(15000, 12000),
            nn.ReLU(),
            nn.Linear(12000, 9000),
            nn.ReLU(),
            nn.Linear(9000, 6000),
            nn.ReLU(),
            nn.Linear(6000, 3000),
            nn.ReLU(),
            nn.Linear(3000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        out = self.layer_stack(x)
        return out


model = Network()
model.to(device)

loss_val=[]
epochs = 20
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
model.train()
for i in range(epochs):
    for data in tqdm(train_loader):
        batch = tuple(t.to(device) for t in data)
        values, labels = batch
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

