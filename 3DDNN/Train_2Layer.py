from Utils import notebook_util
notebook_util.pick_gpu_lowest_memory()

from Network import Network3D_2Layer
from DataManager import DataManager

# 參數
lrArray = [1e-2, 5e-3,  1e-4]
kernelSizeArray = [3, 5, 7, 9]

# 資料集
InputFileList = []
LabeledFileList = []
for i in range(1000):
    InputFileList.append("./Data/Circle_" + str(i) + ".csv")
    LabeledFileList.append("./Data/AnsCircle_" + str(i) + ".png")
DM = DataManager.DataManager(InputFileList, LabeledFileList, 2)

# Network
for lr in lrArray:
    for kernelSize in kernelSizeArray:
        logDir = "./logs/layer2/lr=" + str(lr) + "_kernelSize=" + str(kernelSize)
        net = Network3D_2Layer.Network3D(40, 40, 40, 2, lr, kernelSize, logDir, True)

        # Train
        net.Train(DM, 10000, 128)
        net.SaveWeight(logDir)
        net.Release()
