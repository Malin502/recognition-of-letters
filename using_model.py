import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.onnx
import torch.nn.functional as F

import numpy as np

from PIL import Image




transform = transforms.Compose([
    transforms.RandomAffine([90, 110]),
    transforms.RandomHorizontalFlip(p = 1),
    transforms.ToTensor()
])


test_data = datasets.EMNIST(
    './EMINIST',
    split = 'letters',
    train=False, download=True,
    transform=transform)

batch_size = 64
testdata_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                              shuffle=False, pin_memory=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print("Using device: " + device)

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
        print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
model = NN().to(device)
model.load_state_dict(torch.load("model.pth"))



def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

classes = ['None', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


model.eval
itemnum = 9768

X, y = test_data[itemnum][0], test_data[itemnum][1]

#NNに合うように次元をそろえる
X = X.unsqueeze(dim = 1)


X_show = test_data[itemnum]
img_show(X_show[0].numpy().reshape(28,28)*255)

#予測と正解を表示
with torch.no_grad():
    X = X.to(device)
    pred = model(X)
    print(pred)
            
    predicted, actual =  classes[pred[0].argmax(0)], classes[y]
    print(predicted, actual)
           