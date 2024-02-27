import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.onnx
import torch.nn.functional as F

epochs = 12
batch_size = 64


transform = transforms.Compose([
    transforms.RandomAffine([90, 110]),
    transforms.RandomHorizontalFlip(p = 1),
    transforms.ToTensor()
])



#EMINIStを読み込み
train_data = datasets.EMNIST(
    './EMINIST',
    split='letters',
    train=True, download=True,
    transform=transform)

test_data = datasets.EMNIST(
    './EMINIST',
    split='letters',
    train=False, download=True,
    transform=transform)

traindata_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                               shuffle=True, pin_memory=True)
testdata_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                              shuffle=False, pin_memory=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
#device = "cpu"

print("Using device: " + device)


# ニューラルネットワークの定義
class NN(nn.Module):
    
    def __init__(self):
        super(NN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.MaxPool2d(2))
        
        self.fc1 = nn.Linear(32*7*7, 27)

    def forward(self, x):
        x = x.to(device)
        #print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
model = NN().to(device)

loss_fn =  nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        #print(pred.shape, y.shape)
        
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