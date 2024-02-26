import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.onnx
import torch.nn.functional as F

import cv2
import time
import numpy as np



#Webカメラの設定
DEVICE_ID = 0
WIDTH = 1280 
HEIGHT = 720


#使用するデバイスの設定
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#device = "cuda:0"
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

def main():
    
    model = NN().to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    
    cap = cv2.VideoCapture(DEVICE_ID)
    
    #フォーマット・解像度の設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    # フォーマット・解像度・FPSの取得 
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:{}　width:{}　height:{}".format(fps, width, height))
    
    
    #検出範囲の設定
    size = 50
    x1 = int(width/2 - size)
    x2 = int(width/2 + size)
    y1 = int(height/2 - size)
    y2 = int(height/2 + size)
    
    while True:
        
        #カメラから画像を取得
        _, frame = cap.read()
        if(frame is None):
            continue
        
        img = frame[y1:y2, x1:x2] #検出範囲の画像を取得
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #グレースケールに変換
        
        #EMNISTは文字が白文字なので反転
        img = cv2.bitwise_not(img)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU) #2値化
        img = cv2.resize(img, (28, 28)) #サイズ変更=>28x28
        
        cv2.imshow("img", img)