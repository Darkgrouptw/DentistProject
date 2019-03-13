from Network import Network3D
from DataManager import DataManager

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
# net = Network3D.Network3D(40, 40, 40, 2, lr, kernelSize, logDir, True)
net = Network3D.Network3D(40, 40, 40, 2, lr, kernelSize, logDir, False)

# Train
net.Train(DM, 10000, 128)
net.SaveWeight(logDir)
net.Release()
