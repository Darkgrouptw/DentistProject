from Network.Network import Network
from DataManager import DataManager_OtherSide
import os

# 參數
lr = 5e-3
kernelSize = 9

# 資料集
InputFileList = []
LabeledFileList = []
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
# logDir = "./logs"
# net = Network3D.Network3D(40, 40, 40, 2, lr, kernelSize, logDir, True)
# net = Network(80, 80, 4, lr, kernelSize, logDir, False)

# Train
# net.Train(DM, 10000, 128)
# net.SaveWeight(logDir)
# net.Release()
