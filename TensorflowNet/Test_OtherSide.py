from Network.Network_Prob import Network_Prob
from DataManager import DataManager_OtherSide
import os
import cv2

# 參數
lr = 5e-3
kernelSize = 7

# 資料集
InputFileList = []
LabeledFileList = []
# Windows => E:/DentistData/NetworkData
# Ubuntu => /home/Dark/NetworkData
DataPath = [
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
    "E:/DentistData/NetworkData/2019.04.15 31_1",
    "E:/DentistData/NetworkData/2019.04.15 DONG31",
    "E:/DentistData/NetworkData/2019.04.15 DONG31_2",
    "E:/DentistData/NetworkData/2019.04.15 DONG31_3",
    "E:/DentistData/NetworkData/2019.04.15 DONG36",
    "E:/DentistData/NetworkData/2019.04.15 DONG41",
    # "C:/Users/Dark/Desktop/SourceTree/DentistProject/x64/Release/DentistProjectV2/Images/OCTImages"
]

for i in range(len(DataPath)):
    tempInputPath = os.path.join(DataPath[i], "./OtherSide.png")
    tempLabeledPath = os.path.join(DataPath[i], "./OtherSide_Label.png")

    # 加進陣列裡
    InputFileList.append(tempInputPath)
    LabeledFileList.append(tempLabeledPath)
DM = DataManager_OtherSide.DataManager(InputFileList, LabeledFileList, 1, 101)

# Network
logDir = "./logs/"
# net = Network3D.Network3D(40, 40, 40, 2, lr, kernelSize, logDir, False)
net = Network_Prob(101, 101, 1, lr, kernelSize, logDir, False)

# 讀資料
net.LoadWeight("./logs/OtherSide_kernel_7")

ValidData = DM.TestFirstNBoxOfTrainData(len(DataPath))
predictData = net.Predict(ValidData)
predictData = predictData.reshape([-1, 250, 250])        # 轉成圖
print("Test Image Shape:", predictData.shape)

# 存出來
for i in range(predictData.shape[0]):
    cv2.imwrite("D:/" + str(i) + ".png", predictData[i] * 255)
net.Release()

