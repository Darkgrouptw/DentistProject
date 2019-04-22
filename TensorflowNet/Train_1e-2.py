from Network.Network_Prob import Network_Prob
from DataManager import DataManager
import os

# 參數
# lrArray = [1e-2, 5e-3]
# kernelSizeArray = [3, 5, 7, 9]
lrArray = [5e-3]
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
    "/home/Dark/NetworkData/2019.01.08 ToothBone9.1",
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
    for j in range(StartIndex, EndIndex + 1):
        tempInputPath = os.path.join(DataPath[i], "./boundingBox_v2/" + str(j) + ".png")
        tempLabeledPath = os.path.join(DataPath[i], "./labeled_v2/" + str(j) + ".png")

        if (not os.path.isfile(tempInputPath)) or (not os.path.isfile(tempLabeledPath)):
            ErrorFileList.append(tempInputPath)

        InputFileList.append(tempInputPath)
        LabeledFileList.append(tempLabeledPath)

if len(ErrorFileList) > 0:
    print("以下的檔案有少!!")
    for i in range(len(ErrorFileList)):
        print(ErrorFileList[i])
    assert False
DM = DataManager.DataManager(InputFileList, LabeledFileList, 4, 21)

# Network
for lr in lrArray:
    for kernelSize in kernelSizeArray:
        logDir = "./logs/" + str(lr) + "/kernel_" + str(kernelSize)
        net = Network_Prob(101, 101, 4, lr, kernelSize, logDir, True)

        # Train
        net.Train(DM, 10000, 128)
        net.SaveWeight(logDir)
        net.Release()
