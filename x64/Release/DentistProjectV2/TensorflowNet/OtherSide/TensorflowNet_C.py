# 增加路徑
import os, sys
sys.path.append(os.path.dirname(__file__))

from Network.Network_Prob import Network_Prob
import cv2
import numpy as np

# 設定參數
lr = 1e-4
kernelSize = 7
WindowSize = 101

# Network
logDir = "./TensorflowNet/OtherSide/logs"
net = Network_Prob(WindowSize, WindowSize, 1, lr, kernelSize, logDir, False)

# 讀取 Weight
net.LoadWeight("./TensorflowNet/OtherSide/logs")

#################################################
# Helper Function
#################################################
def _ExtractTheImg(InputImg):
    global WindowSize

    # 先拿圖片資訊
    rows, cols = InputImg.shape[:2]
    DataSize = rows * cols

    # 創建 Input 的 Data
    Data = np.zeros([DataSize, WindowSize, WindowSize], np.float32)

    # 將圖片變大張
    halfRadius = int((WindowSize - 1) / 2)
    LargerInputImg = np.zeros([halfRadius * 2 + rows, halfRadius * 2 + cols], np.float32)
    LargerInputImg[halfRadius:halfRadius + rows, halfRadius:halfRadius + cols] = InputImg

     # 把資料塞回去 InputDataArray
    for rowIndex in range(0, rows):
        for colIndex in range(0, cols):
            # 塞進值
            InputDataTemp = LargerInputImg[rowIndex: rowIndex + WindowSize, colIndex: colIndex + WindowSize]
            DataIndex = rowIndex * cols + colIndex
            Data[DataIndex] = InputDataTemp
    return Data.reshape([-1, WindowSize, WindowSize, 1])

#################################################
# API
#################################################
def PredictImg(img):
    print("圖片大小: ", img.shape)

    # 預測結果
    # predictData = net.Predict(img)
    # predictData = predictData.reshape([-1, 250, 250])

    DataArray = _ExtractTheImg(img)
    print(DataArray.shape)

    predictData = net.Predict(DataArray)
    predictData = predictData.reshape([250, 250]) * 255
    # cv2.imwrite("D:/b.png", predictData)
    print("Python Max: ", np.max(predictData))
    # return predictData
    return  predictData