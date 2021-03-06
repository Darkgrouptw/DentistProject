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
def PredictImg_OtherSide(img):
    print("圖片大小: ", img.shape)

    # Network
    logDir = "./TensorflowNet_C/logs/OtherSide_kernel_7"
    net = Network_Prob(WindowSize, WindowSize, 1, lr, 7, logDir, False)

    # 讀取 Weight
    net.LoadWeight(logDir)
    DataArray = _ExtractTheImg(img)

    # 預測結果
    predictData = net.Predict(DataArray)
    predictData = predictData.reshape([250, 250]) * 255
    net.Release()
    return predictData

def PredictImg_Full(StartIndex, EndIndex, DirPath):
    print("從", StartIndex, "到", EndIndex, "路徑: ", DirPath)

    # Network
    logDir = "./TensorflowNet_C/logs/Full_kernel_5"
    # logDir = "./logs/Full_kernel_5"
    net = Network_Prob(WindowSize, WindowSize, 4, lr, 5, logDir, False)

    # 讀取 Weight
    net.LoadWeight(logDir)
    StartIndex = int(StartIndex)
    EndIndex = int(EndIndex)

    # 預測結果
    for i in range(StartIndex, EndIndex + 1):
        ImgPath = DirPath + "/" + str(i) + ".png"
        SaveImgPath = DirPath + "/Result_" + str(i) + ".png"

        print(ImgPath)
        img = cv2.imread(ImgPath, cv2.IMREAD_GRAYSCALE)
        rows, cols = img.shape

        DataArray = _ExtractTheImg(img)
        predictData = net.Predict(DataArray)
        ImgProb = predictData.reshape([rows, cols, net.OutClass]) * 255

        ImgArgMaxProb = np.argmax(ImgProb, axis=2)
        imgColor = ColorFullPredict[ImgArgMaxProb]

        cv2.imwrite(SaveImgPath, imgColor)
    net.Release()