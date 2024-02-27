import torch
import torch.nn as nn
import torch.onnx
import torch.nn.functional as F

import cv2
import time
import numpy as np



#Webカメラの設定
DEVICE_ID = 1
WIDTH = 1280 
HEIGHT = 720


#使用するデバイスの設定
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#device = "cuda:0"
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
        
        img = img/256.0 #正規化
        img = img[np.newaxis, np.newaxis, :, :] #次元追加 (28,28) => (1,1,28,28)
        pred = model(torch.tensor(img, dtype=torch.float32).to(device))
        
        
        classes = ['None', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
        
        print(pred)
        index = pred[0].argmax(0)
        predicted = classes[index]
        print(predicted)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0))
        cv2.putText(frame, predicted, (10, int(height)-50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA)
        
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()