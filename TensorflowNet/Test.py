from Network.Network_Prob import Network_Prob
from DataManager import DataManager_Test
import os
import numpy as np
import cv2

# 參數
lr = 1e-4
kernelSize = 5

StartIndex = 60
EndIndex = 200

# 資料集
InputFileList = []
LabeledFileList = []
# Windows => E:/DentistData/NetworkData
# Ubuntu => /home/Dark/NetworkData
DataPath = [
    "E:/DentistData/NetworkData/2019.01.08 ToothBone1",
    # "E:/DentistData/NetworkData/2019.01.08 ToothBone2",
    # "E:/DentistData/NetworkData/2019.01.08 ToothBone3.1",
    # "E:/DentistData/NetworkData/2019.01.08 ToothBone3.2",
    # "E:/DentistData/NetworkData/2019.01.08 ToothBone7.1",
    # "E:/DentistData/NetworkData/2019.01.08 ToothBone8.1",
    # "E:/DentistData/NetworkData/2019.01.08 ToothBone9.1",
    # "E:/DentistData/NetworkData/2019.03.05 ToothBone1_slim",
    # "E:/DentistData/NetworkData/2019.03.05 ToothBone2_slim",
    # "E:/DentistData/NetworkData/2019.03.05 ToothBone3_no_slim",
    # "E:/DentistData/NetworkData/2019.03.05 ToothBone4_slim",
    # "E:/DentistData/NetworkData/2019.03.05 ToothBone5_slim",
    # "E:/DentistData/NetworkData/2019.03.05 ToothBone6_slim",
    # "E:/DentistData/NetworkData/2019.03.05 ToothBone7_slim",
    # "E:/DentistData/NetworkData/2019.03.05 ToothBone8_slim"
]

ErrorFileList = []
for i in range(len(DataPath)):
    tempInputArray = []
    tempLabeledArray = []
    for j in range(StartIndex, EndIndex + 1):
        tempInputPath = DataPath[i] + "/boundingBox_v2/" + str(j) + ".png"

        if (not os.path.isfile(tempInputPath)):
            ErrorFileList.append(tempInputPath)

        tempInputArray.append(tempInputPath)

    # 加進去 Array 中
    InputFileList.append(tempInputArray)

if len(ErrorFileList) > 0:
    print("以下的檔案有少!!")
    for i in range(len(ErrorFileList)):
        print(ErrorFileList[i])
    assert False
DM = DataManager_Test.DataManager(InputFileList, 4, 101)

# Network
logDir = "./logs"
net = Network_Prob(101, 101, 4, lr, kernelSize, logDir, False)

# 讀資料
# net.LoadWeight("./logs/0.005/kernel_5")
net.LoadWeight("./logs/Full_kernel_5")

# 存出來
color = np.array(
    [
        [0, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0]
    ],
    dtype = np.uint8
)

# 每一張圖都跑過一次
for i in range(len(DataPath) * 141):
    print(i, "/", len(DataPath) * 141)
    ValidData, rows, cols = DM.TestFullImage(i)
    predictData = net.Predict(ValidData)
    ImgProb = predictData.reshape([rows, cols, net.OutClass])

    # 抓最大的
    ImgArgMaxProb = np.argmax(ImgProb, axis=2)
    imgColor = color[ImgArgMaxProb]
    cv2.imwrite("D:/Data_Slim/" + str(i) + ".png", imgColor)
net.Release()