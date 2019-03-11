from Network import Network3D
from DataManager import DataManager

# Network
# net = Network3D.Network3D(250, 250, 1024, True)
net = Network3D.Network3D(40, 40, 40, 2, True)

# 資料集
InputFileList = []
LabeledFileList = []
for i in range(1000):
    InputFileList.append("./Data/Circle_" + str(i) + ".csv")
    LabeledFileList.append("./Data/AnsCircle_" + str(i) + ".png")
DM = DataManager.DataManager(InputFileList, LabeledFileList, 2)

# Train
net.Train(DM, 10000, 128)
net.SaveWeight()

# Test
# DM.BatchTrainData(1)
# np.set_printoptions(suppress=True, threshold=np.inf)