from Network.Network_Prob import Network_Prob
from DataManager import DataManager_OtherSide
import os

print("ValidTest")

# 參數
# lrArray = [1e-2, 5e-3]
# kernelSizeArray = [3, 5, 7, 9]
lrArray = [1e-4]
kernelSizeArray = [5, 7, 9]

# 資料集
InputFileList = []
LabeledFileList = []
# Windows => E:/DentistData/NetworkData
# Ubuntu => /home/Dark/NetworkData
DataPath = [
    # Valid
    "/home/Dark/NetworkData/31_slim",
    "/home/Dark/NetworkData/32_slim",
    "/home/Dark/NetworkData/41_slim",
    "/home/Dark/NetworkData/42_slim",
    "/home/Dark/NetworkData/43_slim",

    # "/home/Dark/NetworkData/2019.01.08 ToothBone1",
    # "/home/Dark/NetworkData/2019.01.08 ToothBone2",
    # "/home/Dark/NetworkData/2019.01.08 ToothBone3.1",
    # "/home/Dark/NetworkData/2019.01.08 ToothBone3.2",
    # "/home/Dark/NetworkData/2019.01.08 ToothBone7.1",
    # "/home/Dark/NetworkData/2019.01.08 ToothBone8.1",
    # "/home/Dark/NetworkData/2019.01.08 ToothBone9.1",
    # "/home/Dark/NetworkData/2019.03.05 ToothBone1_slim",
    # "/home/Dark/NetworkData/2019.03.05 ToothBone2_slim",
    # "/home/Dark/NetworkData/2019.03.05 ToothBone3_no_slim",
    # "/home/Dark/NetworkData/2019.03.05 ToothBone4_slim",
    # "/home/Dark/NetworkData/2019.03.05 ToothBone5_slim",
    # "/home/Dark/NetworkData/2019.03.05 ToothBone6_slim",
    # "/home/Dark/NetworkData/2019.03.05 ToothBone7_slim",
    # "/home/Dark/NetworkData/2019.03.05 ToothBone8_slim",
    # "/home/Dark/NetworkData/2019.04.15 31_1",
    # "/home/Dark/NetworkData/2019.04.15 DONG31",
    # "/home/Dark/NetworkData/2019.04.15 DONG31_2",
    # "/home/Dark/NetworkData/2019.04.15 DONG31_3",
    # "/home/Dark/NetworkData/2019.04.15 DONG36",
    # "/home/Dark/NetworkData/2019.04.15 DONG41",

    # "E:/DentistData/NetworkData/2019.01.08 ToothBone1",
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
    # "E:/DentistData/NetworkData/2019.03.05 ToothBone8_slim",
    # "E:/DentistData/NetworkData/2019.04.15 31_1",
    # "E:/DentistData/NetworkData/2019.04.15 DONG31",
    # "E:/DentistData/NetworkData/2019.04.15 DONG31_2",
    # "E:/DentistData/NetworkData/2019.04.15 DONG31_3",
    # "E:/DentistData/NetworkData/2019.04.15 DONG36",
    # "E:/DentistData/NetworkData/2019.04.15 DONG41",
]

for i in range(len(DataPath)):
    # 原始
    # tempInputPath = DataPath[i] + "/OtherSide.png"
    # tempLabeledPath = DataPath[i] + "/OtherSide_Label.png"
    tempInputPath = DataPath[i] + "/OtherSideEdit.png"
    tempLabeledPath = DataPath[i] + "/OtherSideEdit_Label.png"

    # 加進陣列裡
    InputFileList.append(tempInputPath)
    LabeledFileList.append(tempLabeledPath)
DM = DataManager_OtherSide.DataManager(InputFileList, LabeledFileList, 1, 101)

# Network
for lr in lrArray:
    for kernelSize in kernelSizeArray:
        logDir = "./logs/OtherSide_kernel_" + str(kernelSize) + "/" + str(lr)
        net = Network_Prob(101, 101, 1, lr, kernelSize, logDir, True)

        # Train
        # assert False
        net.Train(DM, 10000, 128)
        net.SaveWeight(logDir)
        net.Release()
