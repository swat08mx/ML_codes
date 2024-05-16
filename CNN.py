import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_curve, auc, precision_score, confusion_matrix, accuracy_score
from sklearn.linear_model import Lasso
from torch import nn
import statistics
import numpy as np
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

data1 = pd.read_csv("gxp_dataset.csv")
temp = pd.DataFrame(data1['label'].to_list(), columns=['labels'])
final = pd.read_csv("lasso_big_dataset.csv")
final = final[:-7]
temp = temp[:-7]

vals = []
for i in range(1):
    temp = pd.DataFrame(temp, columns=['labels'])
    X_train, X_test, y_train, y_test = train_test_split(final, temp, test_size=0.3, shuffle=True, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_train_tensor = torch.tensor(X_train)
    X_test_tensor = torch.tensor(X_test)
    y_train_tensor = torch.tensor(y_train.values)
    y_test_tensor = torch.tensor(y_test.values)

    train_set = TensorDataset(X_train_tensor, y_train_tensor)
    test_set = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_set, batch_size=12)
    test_loader = DataLoader(test_set, batch_size=12)


    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=1, stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(32, 12, kernel_size=3, padding=1)
            #  self.fc1 = nn.Linear(1178, 128)
            #  self.fc2 = nn.Linear(128, 300)
            #  self.fc3 = nn.Linear(300, 128)
            #  self.fc4 = nn.Linear(128, 60)
            self.fc5 = nn.Linear(53, 25)
            self.fc6 = nn.Linear(25, 2)
            self.dropout = nn.Dropout(0.8)

        def forward(self, x):
            x = x.unsqueeze(0)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            #  x = F.relu(self.fc1(x))
            #  x = F.relu(self.fc2(x))
            #  x = F.relu(self.fc3(x))
            #  x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x = self.fc6(x)
            return x


    model = CNN()
    model.to(device)
    # loss_val = []
    # epochs = 200
    # running_loss = []
    # optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
    # loss_fn = nn.CrossEntropyLoss()
    # model.train()
    # for i in range(epochs):
    #     for data in tqdm(train_loader):
    #         batch = tuple(t.to(device) for t in data)
    #         values, labels = batch
    #         output = model(values.float())
    #         print(f"Output: {output.shape}")
    #         print(f"Output_maxed:{torch.argmax(output, dim=1)}")
    #         # labels = labels.view(-1)
    #         print(f"Labels: {labels.shape}")
    #         labels = torch.squeeze(labels)
    #         loss = loss_fn(output, labels)
    #         # loss.requires_grad=True
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         running_loss.append(loss.item())
    #         # print(f"Loss is :{loss.item()}")
    #     print(f"Epoch {i}")
    #     total = 0
    #     for i in range(len(running_loss)): total += running_loss[i]
    #     loss_val.append(total / len(running_loss))
    #     print(f"Training Loss for this epoch is: {total / len(running_loss)}")
    #
    # PATH = "model_CNN.pth"
    # torch.save(model.state_dict(), PATH)

    model.load_state_dict(torch.load('model_CNN.pth'))
    model.eval()
    correct = 0
    total = 0
    pred = []
    label = []
    prob=[]
    for data in tqdm(test_loader):
        batch = tuple(t.to(device) for t in data)
        values, labels = batch
        output = model(values.float())
        probabilities = F.softmax(output, dim=1)[:, 1]
        predicted = torch.argmax(output, dim=1)
        print(f"Predicted {predicted.shape}")
        print(f"Labels {labels.shape}")
        prob.append(probabilities.cpu())
        pred.append(predicted)
        label.append(labels)
        total += labels.size(0)
    #     print(len(pred))
    #     print(len(label))
    #     correct += (predicted == labels).sum().item()
    # accuracy = (correct/total)*100
    # vals.append(accuracy)
    # print(f"\nTest Accuracy: {format(accuracy, '.4f')}%\n")

    var = []
    for i in range(len(pred)):
        var.append(pred[i].tolist())
    pred_new = []
    for j in range(len(var)):
        for k in range(len(var[j])):
            pred_new.append(var[j][k])
    var_new = []
    for i in range(len(label)):
        var_new.append(label[i].tolist())
    label_new = []
    for j in range(len(var_new)):
        for k in range(len(var[j])):
            label_new.append(var_new[j][0])
    var = []
    for i in range(len(prob)):
        var.append(prob[i].tolist())
    prob_new = []
    for j in range(len(var)):
        for k in range(len(var[j])):
            prob_new.append(var[j][k])

    print(confusion_matrix(label_new, pred_new))
    print(f"Accuracy is: {accuracy_score(label_new, pred_new)}")
    vals.append(accuracy_score(label_new, pred_new))

    cm = confusion_matrix(label_new, pred_new)
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=['Autism', 'Control'], yticklabels=['Autism', 'Control'])
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    # plt.title("Confusion Matrix for Feedforward network", fontsize=17)
    plt.show()
    fpr, tpr, _ = roc_curve(label_new, prob_new)
    curve = auc(fpr, tpr)
    print(f"AUC score is: {curve}")
    print(f"Precision score is: {precision_score(label_new, pred_new)}")
    print(f"F1 score is: {f1_score(label_new, pred_new)}")
print(vals)
