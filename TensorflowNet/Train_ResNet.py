from Network.Network_Resnet import Network_ResNet
from DataManager import DataManager
import os

# 參數
lrArray = [1e-4]
kernelSizeArray = [5]


StartIndex = 60
EndIndex = 200

# 資料集
InputFileList = []
LabeledFileList = []
DataPath = [
    "/home/Dark/NetworkData/2019.01.08 ToothBone1",
    "/home/Dark/NetworkData/2019.01.08 ToothBone2",
    "/home/Dark/NetworkData/2019.01.08 ToothBone3.1",
    "/home/Dark/NetworkData/2019.01.08 ToothBone3.2",
    "/home/Dark/NetworkData/2019.01.08 ToothBone7.1",
    "/home/Dark/NetworkData/2019.01.08 ToothBone8.1",
    # "E:/DentistData/NetworkData/2019.01.08 ToothBone1",
    # "E:/DentistData/NetworkData/2019.01.08 ToothBone2",
    # "E:/DentistData/NetworkData/2019.01.08 ToothBone3.1",
    # "E:/DentistData/NetworkData/2019.01.08 ToothBone3.1",
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
        tempLabeledPath = DataPath[i] + "/labeled_v2/" + str(j) + ".png"

        if (not os.path.isfile(tempInputPath)) or (not os.path.isfile(tempLabeledPath)):
            ErrorFileList.append(tempInputPath)

        tempInputArray.append(tempInputPath)
        tempLabeledArray.append(tempLabeledPath)

    # 加進去 Array 中
    InputFileList.append(tempInputArray)
    LabeledFileList.append(tempLabeledArray)

if len(ErrorFileList) > 0:
    for i in range(len(ErrorFileList)):
        print(ErrorFileList[i])
    print("以下的檔案有少!!")
    assert False
DM = DataManager.DataManager(InputFileList, LabeledFileList, 4, 101)

# Network
for lr in lrArray:
    for kernelSize in kernelSizeArray:
        logDir = "./logs/ResNet_Full_kernel_" + str(kernelSize)
        net = Network_ResNet(101, 101, 4, lr, kernelSize, logDir, True)

        # Train
        net.Train(DM, 10000, 128)
        net.SaveWeight(logDir)
        net.Release()