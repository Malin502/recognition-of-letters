import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.onnx
import torch.nn.functional as F

epochs = 15
batch_size = 64

#EMINIStを読み込み
train_data = datasets.EMNIST(
    './EMINIST',
    split='letters',
    train=True, download=True,
    transform=transforms.ToTensor())

test_data = datasets.EMNIST(
    './EMINIST',
    split='letters',
    train=False, download=True,
    transform=transforms.ToTensor())

traindata_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                               shuffle=True, pin_memory=True)
testdata_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                              shuffle=True, pin_memory=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
#device = "cuda:0"

print("Using device: " + device)


# ニューラルネットワークの定義
class NN(nn.Module):
    
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128*7*7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 26)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 128*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
model = NN().to(device)

loss_fn =  nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") 
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
#学習と検証
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(traindata_loader, model, loss_fn, optimizer)
    test(testdata_loader, model, loss_fn)

print("Done!")
        
#モデルの保存
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
