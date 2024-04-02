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


# data1 = pd.read_csv("sample.csv")
# final=data1
# dim=120
# labels_two = final['A/C'].to_list()
# temp=[]
# for i in range(len(labels_two)):
#     if labels_two[i] == 'A':
#         temp.append(1)
#     else:
#         temp.append(0)
# #temp = pd.DataFrame(temp, columns=['labels'])
# final.drop('A/C', axis=1, inplace=True)
# #final.drop('Sample_ID', axis=1, inplace=True)
# from sklearn.preprocessing import StandardScaler
# x = data1.loc[:, data1.columns].values
# x = StandardScaler().fit_transform(x) # normalizing the features
# import numpy as np
# np.mean(x),np.std(x)
# feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
# normalised = pd.DataFrame(x,columns=feat_cols)
# normalised.tail()
# from sklearn.decomposition import PCA
# pca = PCA(n_components=dim)
# principalComponents = pca.fit_transform(x)
# cols = ['PC'+str(i) for i in range(dim)]
# principal_Df = pd.DataFrame(data = principalComponents, columns = cols)
# principal_Df.head()
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
# calc = pca.explained_variance_ratio_
# calc.sum()
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

data = pd.read_csv("sample.csv")
#data=data.iloc[:, :9000]
print(data)
temp = []
for i in range(len(data['A/C'])):
    if data['A/C'][i] == 'A':
        temp.append(1)
    else:
        temp.append(0)
data.drop('A/C', axis=1, inplace=True)
#data.drop('Sample_ID', axis=1, inplace=True)
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

train_loader = DataLoader(train_set, batch_size=3)
test_loader = DataLoader(test_set, batch_size=3)


class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=1, stride=2)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
    self.fc5 = nn.Linear(2614, 100)
    # self.fc6 = nn.Linear(1000, 100)
    # self.fc7 = nn.Linear(500, 250)
    # self.fc8 = nn.Linear(300, 150)
    # self.fc9 = nn.Linear(150, 70)
    # self.fc10 = nn.Linear(70, 30)
    # self.fc11 = nn.Linear(30, 15)
    # self.fc12 = nn.Linear(250, 100)
    self.fc13 = nn.Linear(100, 50)
    self.fc14 = nn.Linear(50, 8)
    self.fc15 = nn.Linear(8, 2)
    self.dropout = nn.Dropout(0.7)

  def forward(self, x):
    x = x.unsqueeze(0)
    x = F.relu(self.conv1(x))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = self.pool(F.relu(self.conv4(x)))
    x = torch.flatten(x, 1)
    x = self.dropout(x)
    x = F.relu(self.fc5(x))
    # x = F.relu(self.fc6(x))
    # x = F.relu(self.fc7(x))
    # x = F.relu(self.fc8(x))
    # x = F.relu(self.fc9(x))
    # x = F.relu(self.fc10(x))
    # x = F.relu(self.fc11(x))
    # x = F.relu(self.fc12(x))
    x = F.relu(self.fc13(x))
    x = F.relu(self.fc14(x))
    x = self.fc15(x)
    return x

model = CNN()
model.to(device)

running_loss=[]
loss_val=[]
epochs = 200
optimizer = optim.Adam(params=model.parameters(), lr=0.00001)
loss_fn = nn.CrossEntropyLoss()
model.train()
for i in range(epochs):
    for data in tqdm(train_loader):
        batch = tuple(t.to(device) for t in data)
        values, labels = batch
        output = model(values.float())
        # print(f"Input: {values.shape}")
        maxed = torch.argmax(output, dim=1)
        # print(f"Output_maxed:{torch.argmax(output, dim=1)}")
        #print(f"Labels: {labels.shape}")
        loss = loss_fn(output, labels)
        #loss.requires_grad=True
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss.append(loss.item())
    print(f"Epoch {i}")
    total=0
    for i in range(len(running_loss)): total+=running_loss[i]
    loss_val.append(total/len(running_loss))
    print(f"Training Loss for this epoch is: {total/len(running_loss)}")


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
    # print(f"Predicted {predicted}")
    # print(f"Labels {labels}")
    pred.append(predicted.cpu())
    label.append(labels.cpu())
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print(correct)
print(total)
accuracy = (correct/total)*100
print(f"\nTest Accuracy: {format(accuracy, '.4f')}%\n")

var = []
for i in range(len(pred)):
  var.append(pred[i].tolist())
pred_new=[]
for j in range(len(var)):
  for k in range(len(var[j])):
    pred_new.append(var[j][k])
var_new = []
for i in range(len(label)):
  var_new.append(label[i].tolist())
label_new=[]
for j in range(len(var_new)):
  for k in range(len(var[j])):
    label_new.append(var_new[j][0])
from sklearn.metrics import confusion_matrix, precision_score
print(confusion_matrix(label_new, pred_new))
print(f"Precision is: {precision_score(label_new, pred_new)}")
