

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch import nn
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
import statistics

data1 = pd.read_csv("final_data.csv")
temp = []
for i in range(len(data1['A/C'])):
    if data1['A/C'][i] == 'A':
        temp.append(1)
    else:
        temp.append(0)
data1.drop(['A/C', 'Sample_ID'], axis=1, inplace=True)
temp = pd.DataFrame(temp, columns=['labels'])
X_train, X_test, y_train, y_test = train_test_split(data1, temp, test_size=0.3, shuffle=True)
df_new = pd.read_csv("lasso_dataset.csv")




# lasso = Lasso()

# names = data1.columns
# print("Column Names: {}".format(names.values))


# lasso1 = Lasso(alpha=0.001)
# lasso1.fit(X_train, y_train)
# lasso1_coef = np.abs(lasso1.coef_)
# lists = lasso1_coef.tolist()
# print(min(lists))
# print(statistics.median(lists))
# median = statistics.median(lists)
# print(len(lasso1_coef))

# feature_subset=np.array(names)[lasso1_coef>median]
# print("Selected Feature Columns: {}".format(feature_subset))

# df_new = data1[feature_subset]
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(df_new)
vals=[]
for i in range(10):
  temp = pd.DataFrame(temp, columns=['labels'])
  X_train, X_test, y_train, y_test = train_test_split(df_new, temp, test_size=0.3, shuffle=True)
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  X_train_tensor = torch.tensor(X_train)
  X_test_tensor = torch.tensor(X_test)
  y_train_tensor = torch.tensor(y_train.values)
  y_test_tensor = torch.tensor(y_test.values)

  train_set = TensorDataset(X_train_tensor, y_train_tensor)
  test_set = TensorDataset(X_test_tensor, y_test_tensor)

  train_loader = DataLoader(train_set, batch_size=32)
  test_loader = DataLoader(test_set, batch_size=1)

  class Network(nn.Module):
      def __init__(self):
          super().__init__()
          self.layer_stack = nn.Sequential(
            #   nn.Linear(187, 400),
            #   nn.ReLU(),
            #   nn.Linear(400, 200),
            #   nn.ReLU(),
              nn.Linear(187, 100),
              nn.ReLU(),
              nn.Dropout(0.8),
              nn.Linear(100, 50),
              nn.ReLU(),
              nn.Linear(50, 20),
              nn.ReLU(),
              nn.Linear(20, 2),
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
          print(f"Output:{torch.argmax(output, dim=1)}")
          #print(f"targets: {labels}")
          labels = torch.squeeze(labels)
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
  vals.append(accuracy)
print(vals)



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
from sklearn.metrics import confusion_matrix
print(confusion_matrix(label_new, pred_new))

