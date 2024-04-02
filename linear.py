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
data1 = pd.read_csv("sample.csv")
final=data1
dim=120
labels_two = final['A/C'].to_list()
temp=[]
for i in range(len(labels_two)):
    if labels_two[i] == 'A':
        temp.append(1)
    else:
        temp.append(0)
final.drop('A/C', axis=1, inplace=True)
#final.drop('Sample_ID', axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler
x = data1.loc[:, data1.columns].values
x = StandardScaler().fit_transform(x) # normalizing the features
import numpy as np
np.mean(x),np.std(x)
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised = pd.DataFrame(x,columns=feat_cols)
normalised.tail()
from sklearn.decomposition import PCA
pca = PCA(n_components=dim)
principalComponents = pca.fit_transform(x)
cols = ['PC'+str(i) for i in range(dim)]
principal_Df = pd.DataFrame(data = principalComponents, columns = cols)
principal_Df.head()
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
calc = pca.explained_variance_ratio_
print(calc.sum())

print(principal_Df.head())
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
# data1 = pd.read_csv("/content/drive/MyDrive/final_data.csv")
# temp_new = data1['A/C']
# #data1=data1.iloc[:, :9000]
# print(data1)
# temp = []
# for i in range(len(data1['A/C'])):
#     if data1['A/C'][i] == 'A':
#         temp.append(1)
#     else:
#         temp.append(0)
# data1.drop('A/C', axis=1, inplace=True)
# data1.drop('Sample_ID', axis=1, inplace=True)
# # temp = pd.DataFrame(temp, columns=['labels'])
# final=data1


X_train, X_test, y_train, y_test = train_test_split(principal_Df, temp, test_size=0.3, shuffle=True)
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
            nn.Linear(120, 100),
            nn.ReLU(),
            nn.Linear(100, 80),
            nn.ReLU(),
            nn.Linear(80, 60),
            nn.ReLU(),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 8),
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
running_loss=[]
epochs = 200
optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
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
