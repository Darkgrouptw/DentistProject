from Network import Network3D
from DataManager import DataManager

# 參數
lr = 1e-2
kernelSize = 9

# 資料集
InputFileList = []
LabeledFileList = []
for i in range(1000):
    InputFileList.append("./Data/Circle_" + str(i) + ".csv")
    LabeledFileList.append("./Data/AnsCircle_" + str(i) + ".png")
DM = DataManager.DataManager(InputFileList, LabeledFileList, 2)

# Network
logDir = "./logs/layer3/lr=" + str(lr) + "_kernelSize=" + str(kernelSize)
net = Network3D.Network3D(40, 40, 40, 2, lr, kernelSize, logDir, True)

# Train
net.Train(DM, 10000, 128)
net.SaveWeight(logDir)
net.Release()
