from Network.Network import Network
from DataManager import DataManager_OtherSide
import os

# 參數
lrArray = [1e-2, 5e-3]
kernelSizeArray = [3, 5, 7, 9]

# 資料集
InputFileList = []
LabeledFileList = []
# Windows => E:/DentistData/NetworkData
# Ubuntu => /home/Dark/NetworkData
DataPath = [
    "/home/Dark/NetworkData/2019.01.08 ToothBone1",
    "/home/Dark/NetworkData/2019.01.08 ToothBone2",
    "/home/Dark/NetworkData/2019.01.08 ToothBone3.1",
    "/home/Dark/NetworkData/2019.01.08 ToothBone3.2",
    "/home/Dark/NetworkData/2019.01.08 ToothBone7.1",
    "/home/Dark/NetworkData/2019.01.08 ToothBone8.1",
    "/home/Dark/NetworkData/2019.01.08 ToothBone9.1",
    "/home/Dark/NetworkData/2019.03.05 ToothBone1_slim",
    "/home/Dark/NetworkData/2019.03.05 ToothBone2_slim",
    "/home/Dark/NetworkData/2019.03.05 ToothBone3_no_slim",
    "/home/Dark/NetworkData/2019.03.05 ToothBone4_slim",
    "/home/Dark/NetworkData/2019.03.05 ToothBone5_slim",
    "/home/Dark/NetworkData/2019.03.05 ToothBone6_slim",
    "/home/Dark/NetworkData/2019.03.05 ToothBone7_slim",
    "/home/Dark/NetworkData/2019.03.05 ToothBone8_slim"
]

for i in range(len(DataPath)):
    tempInputPath = os.path.join(DataPath[i], "./OtherSide.png")
    tempLabeledPath = os.path.join(DataPath[i], "./OtherSide_Label.png")

    # 加進陣列裡
    InputFileList.append(tempInputPath)
    LabeledFileList.append(tempLabeledPath)
DM = DataManager_OtherSide.DataManager(InputFileList, LabeledFileList, 1, 101, 1)

# Network
logDir = "./logs"

for lr in lrArray:
    for kernelSize in kernelSizeArray:
        net = Network(101, 101, 1, lr, kernelSize, logDir, False)

        # Train
        net.Train(DM, 10000, 128)
        net.SaveWeight(logDir)
        net.Release()
