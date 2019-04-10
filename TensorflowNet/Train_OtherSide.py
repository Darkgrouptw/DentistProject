from Network.Network_Prob import Network_Prob
from DataManager import DataManager_OtherSide
import os

# 參數
# lrArray = [1e-2, 5e-3]
# kernelSizeArray = [3, 5, 7, 9]
lrArray = [5e-3]
kernelSizeArray = [5]

# 資料集
InputFileList = []
LabeledFileList = []
# Windows => E:/DentistData/NetworkData
# Ubuntu => /home/Dark/NetworkData
DataPath = [
    "E:/DentistData/NetworkData/2019.01.08 ToothBone1",
    "E:/DentistData/NetworkData/2019.01.08 ToothBone2",
    "E:/DentistData/NetworkData/2019.01.08 ToothBone3.1",
    "E:/DentistData/NetworkData/2019.01.08 ToothBone3.2",
    "E:/DentistData/NetworkData/2019.01.08 ToothBone7.1",
    "E:/DentistData/NetworkData/2019.01.08 ToothBone8.1",
    "E:/DentistData/NetworkData/2019.01.08 ToothBone9.1",
    "E:/DentistData/NetworkData/2019.03.05 ToothBone1_slim",
    "E:/DentistData/NetworkData/2019.03.05 ToothBone2_slim",
    "E:/DentistData/NetworkData/2019.03.05 ToothBone3_no_slim",
    "E:/DentistData/NetworkData/2019.03.05 ToothBone4_slim",
    "E:/DentistData/NetworkData/2019.03.05 ToothBone5_slim",
    "E:/DentistData/NetworkData/2019.03.05 ToothBone6_slim",
    "E:/DentistData/NetworkData/2019.03.05 ToothBone7_slim",
    "E:/DentistData/NetworkData/2019.03.05 ToothBone8_slim"
]

for i in range(len(DataPath)):
    tempInputPath = os.path.join(DataPath[i], "./OtherSide.png")
    tempLabeledPath = os.path.join(DataPath[i], "./OtherSide_Label.png")

    # 加進陣列裡
    InputFileList.append(tempInputPath)
    LabeledFileList.append(tempLabeledPath)
DM = DataManager_OtherSide.DataManager(InputFileList, LabeledFileList, 1, 101)

# Network
for lr in lrArray:
    for kernelSize in kernelSizeArray:
        logDir = "./logs/" + str(lr) + "/kernel_" + str(kernelSize)
        net = Network_Prob(101, 101, 1, lr, kernelSize, logDir, True)

        # Train
        # assert False
        net.Train(DM, 10000, 128)
        net.SaveWeight(logDir)
        net.Release()