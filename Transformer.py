import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import nn
import pandas as pd
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
# device = torch.device("cpu")
print(device)
data1 = pd.read_csv("final_data.csv")
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
vector=[]
for i in range(len(final)):
  vector.append(final.iloc[i].tolist())
X_train, X_test = train_test_split(vector, test_size=0.3, shuffle=True)
X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)

train_set = TensorDataset(X_train_tensor)
test_set = TensorDataset(X_test_tensor)

train_loader = DataLoader(train_set, batch_size=10)
test_loader = DataLoader(test_set, batch_size=10)

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

ntokens = 18836  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

loss_fn = nn.CrossEntropyLoss()
lr = 0.001
bptt = len(vector[0])
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target
optimizer = optim.SGD(model.parameters(), lr=lr)
model.train()
log_interval=200
loss_val_train=[]
running_loss_train=[]
epochs = 30
model.train()
j=0
for i in range(epochs):
    for data in tqdm(train_loader):
        batch = tuple(t.to(device) for t in data)
        values = batch
        datas = torch.tensor(values[0])
        data, targets = get_batch(datas, 0)
        output = model(data.int())
        output_flat = output.view(-1, ntokens)
        loss = loss_fn(output_flat, targets.type(torch.LongTensor).to(device))
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        running_loss_train.append(loss.item())
    print(f"Epoch {i}")
    total=0
    for i in range(len(running_loss_train)): total+=running_loss_train[i]
    loss_val_train.append(total/len(running_loss_train))
    print(f"Training Loss for this epoch is: {total/len(running_loss_train)}")

model.eval()
total=0
running_loss_test=[]
with torch.no_grad():
  for data in tqdm(test_loader):
    batch = tuple(t.to(device) for t in data)
    values = batch
    datas = torch.tensor(values[0])
    data, targets = get_batch(datas, 0)
    output = model(data.int())
    output_flat = output.view(-1, ntokens)
    loss = loss_fn(output_flat, targets.type(torch.LongTensor).to(device))
    running_loss_test.append(loss.item())
  for i in range(len(running_loss_test)): total+=running_loss_test[i]
  print(f"Total test loss is: {total/len(running_loss_test)}")
    running_loss_test.append(loss.item())
  for i in range(len(running_loss_test)): total+=running_loss_test[i]
  print(f"Total test loss is: {total/len(running_loss_test)}")
