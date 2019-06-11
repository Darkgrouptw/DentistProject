﻿# 增加路徑
import os, sys
sys.path.append(os.path.dirname(__file__))

from Network.Network_Prob import Network_Prob
import numpy as np
import cv2

# 設定參數
lr = 5e-3
WindowSize = 101

# 設定圖片預測大小
ColorFullPredict = np.array(
    [
        [0, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0]
    ],
    dtype = np.uint8
)

# 網路相關變數
net_Full = 0
net_OtherSide = 0



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
    LargerInputImg[halfRadius:halfRadius + rows, halfRadius:halfRadius + cols] = InputImg / 255

     # 把資料塞回去 InputDataArray
    for rowIndex in range(rows):
        for colIndex in range(cols):
            # 塞進值
            InputDataTemp = LargerInputImg[rowIndex: rowIndex + WindowSize, colIndex: colIndex + WindowSize]
            DataIndex = rowIndex * cols + colIndex
            Data[DataIndex] = InputDataTemp
    return Data.reshape([-1, WindowSize, WindowSize, 1])

#################################################
# API
#################################################
# 初始化
def InitTensorflow():
    global net_Full, net_OtherSide

    # Network_Full
    logDir = "./logs/Full_kernel_5"
    net_Full = Network_Prob(WindowSize, WindowSize, 4, lr, 5, logDir, False)
    net_Full.LoadWeight(logDir)

    # Network_OtherSide
    logDir = "./logs/OtherSide_kernel_7"
    net_OtherSide = Network_Prob(WindowSize, WindowSize, 1, lr, 7, logDir, False)
    net_OtherSide.LoadWeight(logDir)

# 預測單張
def PredictImg_OtherSide(img):
    # 拆圖片
    DataArray = _ExtractTheImg(img)

    # 預測結果
    predictData = net_OtherSide.Predict(DataArray)
    predictData = predictData.reshape([250, 250]) * 255
    return predictData

# 預測全部
def PredictImg_Full(imgs):
    # 預測結果
    result = []
    for i in range(len(imgs)):
        rows, cols = imgs[i].shape

        DataArray = _ExtractTheImg(imgs[i])
        predictData = net_Full.Predict(DataArray)
        ImgProb = predictData.reshape([rows, cols, net_Full.OutClass]) * 255

        ImgArgMaxProb = np.argmax(ImgProb, axis=2)
        imgColor = ColorFullPredict[ImgArgMaxProb]

        # 回傳圖片
        result.append(imgColor)
    return result

# 刪除
def Release():
    global net_Full, net_OtherSide
    net_Full.Release()
    net_OtherSide.Release()