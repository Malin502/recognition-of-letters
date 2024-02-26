import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.onnx
import torch.nn.functional as F



test_data = datasets.EMNIST(
    './EMINIST',
    split = 'letters',
    train=False, download=True,
    transform=transforms.ToTensor())

batch_size = 64
testdata_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                              shuffle=True, pin_memory=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device: " + device)

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
model.load_state_dict(torch.load("model.pth"))

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


model.eval
itemnum = 3452

X, y = test_data[itemnum][0], test_data[itemnum][1]
with torch.no_grad():
    X = X.to(device)
    pred = model(X)
    print(pred)
            
    predicted, actual =  classes[pred[0].argmax(0)], classes[y]
    print(predicted, actual)
           
