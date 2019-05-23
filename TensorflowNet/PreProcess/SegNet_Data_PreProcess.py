import numpy as np
import cv2
from tqdm import tqdm
import os

# 讀取資料
DataPath = [
    "E:/DentistData/NetworkData/2019.01.08 ToothBone1",
    "E:/DentistData/NetworkData/2019.01.08 ToothBone2",
    "E:/DentistData/NetworkData/2019.01.08 ToothBone3.1",
    "E:/DentistData/NetworkData/2019.01.08 ToothBone3.2",
    "E:/DentistData/NetworkData/2019.01.08 ToothBone7.1",
    "E:/DentistData/NetworkData/2019.01.08 ToothBone8.1",
]
StartIndex = 60
EndIndex = 200

InputFileList = []
LabeledFileList = []
BoundingBoxFileList = []
SegNetDirList = []

ErrorFileList = []
for i in range(len(DataPath)):
    tempInputArray = []
    tempLabeledArray = []
    for j in range(StartIndex, EndIndex + 1):
        tempInputPath = DataPath[i] + "/origin_v2/" + str(j) + ".png"
        tempLabeledPath = DataPath[i] + "/labeled_v2/" + str(j) + ".png"

        if (not os.path.isfile(tempInputPath)) or (not os.path.isfile(tempLabeledPath)):
            ErrorFileList.append(tempInputPath)

        tempInputArray.append(tempInputPath)
        tempLabeledArray.append(tempLabeledPath)

    # 判斷有沒有存在，創建資料
    SegNetDir = DataPath[i] + "/SegNet_Labeled"
    SegNetDirList.append(SegNetDir)
    if (not os.path.isdir(SegNetDir)):
        os.mkdir(SegNetDir)

    # 加進去 Array 中
    InputFileList.append(tempInputArray)
    LabeledFileList.append(tempLabeledArray)
    BoundingBoxFileList.append(DataPath[i] + "/boundingBox.txt")

# Error 問題
if len(ErrorFileList) > 0:
    print("以下的檔案有少!!")
    for i in range(len(ErrorFileList)):
        print(ErrorFileList[i])
    assert False

# 接著去拿資料貼回原圖
for i in range(len(DataPath)):
    print(i, "/", len(DataPath))

    # 開啟 Bounding Box txt
    BoundingData = open(BoundingBoxFileList[i], 'r').read().replace("\r\n", "\n")
    BoundingData = BoundingData.split("\n")[1: - 1]         # 最前面的 Title & 最後面的空的不要
    print(len(BoundingData), BoundingData)

    for j in range(len(BoundingData)):
        # 抓取 Bounding Box 的資訊
        BoundingInfo = BoundingData[j].split(" ")
        assert len(BoundingData) != 4, "確定有四筆資料"
        lfX = int(BoundingInfo[0])
        lfY = int(BoundingInfo[1])
        rbX = int(BoundingInfo[2])
        rbY = int(BoundingInfo[3])

        # 將資料塞回大圖
        LargerImage = np.zeros([250, 1024], np.uint8)
        assert ((rbY - lfY) == )and, "確定有四筆資料"

        LabeledImage = cv2.imread(LabeledFileList[i][j])
        LargerImage[lfY:rbY, lfX:rbX] = LabeledImage

        assert False
