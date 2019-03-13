from Network import Network3D
from DataManager import DataManager
import cv2

# 參數
lr = 5e-3
kernelSize = 9

# 資料集
InputFileList = []
LabeledFileList = []
for i in range(1000):
    InputFileList.append("./Data/Circle_" + str(i) + ".csv")
    LabeledFileList.append("./Data/AnsCircle_" + str(i) + ".png")
DM = DataManager.DataManager(InputFileList, LabeledFileList, 2)

# Network
logDir = "./logs"
net = Network3D.Network3D(40, 40, 40, 2, lr, kernelSize, logDir, False)

# 讀資料
net.LoadWeight("./logs")

ValidData = DM.TestFirstNBoxOfValidData(3)
predictData = net.Predict(ValidData)

# 存出來
for i in range(predictData.shape[0]):
    cv2.imwrite("D:/" + str(i) + ".png", predictData[i])
net.Release()

