from Network import Network3D
from DataManager import DataManager
import numpy as np


# Network
# net = Network3D.Network3D(250, 250, 1024, True)
net = Network3D.Network3D(40, 40, 40, True)

# 資料集
InputFileList = []
LabeledFileList = []
for i in range(1000):
    InputFileList.append("./Data/Circle_" + str(i) + ".csv")
    LabeledFileList.append("./Data/AnsCircle_" + str(i) + ".png")
DM = DataManager.DataManager(InputFileList, LabeledFileList, 2)

TrainData, LabeledData = DM.BatchTrainData()
net.LoadWeight()

predictData =  net.Predict(TrainData[0:1])
predictImg = np.argmax(predictData, axis=3).reshape(40, 40) * 255


