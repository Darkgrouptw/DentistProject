from Network.Network import Network
from DataManager import DataManager
import os

# 參數
lr = 5e-3
kernelSize = 9
StartIndex = 60
EndIndex = 200

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
    for j in range(StartIndex, EndIndex + 1):
        tempInputPath = os.path.join(DataPath[i], "./boundingBox_v2/" + str(j) + ".png")
        tempLabeledPath = os.path.join(DataPath[i], "./labeled_v2/" + str(j) + ".png")
    #     InputFileList.append(tempPath)
    #     LabeledFileList.append("./Data/AnsCircle_" + str(i) + ".png")
    # break
    # InputFileList.append("./Data/Circle_" + str(i) + ".csv")
    # LabeledFileList.append("./Data/AnsCircle_" + str(i) + ".png")
# DM = DataManager.DataManager(InputFileList, LabeledFileList, 2)

# Network
# logDir = "./logs"
# net = Network3D.Network3D(40, 40, 40, 2, lr, kernelSize, logDir, True)
# net = Network(80, 80, 4, lr, kernelSize, logDir, False)

# Train
# net.Train(DM, 10000, 128)
# net.SaveWeight(logDir)
# net.Release()
