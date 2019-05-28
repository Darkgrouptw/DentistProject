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
SegNetInputDirList = []
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
    SegNetInputDir = DataPath[i] + "/SegNet_Input"
    SegNetInputDirList.append(SegNetInputDir)
    SegNetDir = DataPath[i] + "/SegNet_Labeled"
    SegNetDirList.append(SegNetDir)
    if (not os.path.isdir(SegNetInputDir)):
        os.mkdir(SegNetInputDir)
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
if True:
    for i in range(len(InputFileList)):
        print(i, "/", len(InputFileList))

        # 開啟 Bounding Box txt
        BoundingData = open(BoundingBoxFileList[i], 'r').read().replace("\r\n", "\n")
        BoundingData = BoundingData.split("\n")[1: - 1]         # 最前面的 Title & 最後面的空的不要
        # print(len(BoundingData), BoundingData)

        for j in tqdm(range(len(InputFileList[i]))):
            # 抓取 Bounding Box 的資訊
            BoundingInfo = BoundingData[StartIndex + j].split(" ")
            assert len(BoundingInfo) == 4, "確定有四筆資料"
            lfX = int(BoundingInfo[0])
            lfY = int(BoundingInfo[1])
            rbX = int(BoundingInfo[2])
            rbY = int(BoundingInfo[3])

            # 讀原圖
            LabeledImage = cv2.imread(LabeledFileList[i][j])
            rows, cols = LabeledImage.shape[:2]

            # 將資料塞回大圖
            # assert (((rbY - lfY) == rows) and ((rbX - lfX) == cols)), "圖片大小不正確"
            LargerImage = np.zeros([250, 1024], np.uint8)

            # 將資料轉成灰階
            for rowIndex in range(rows):
                for colIndex in range(cols):
                    result = 0
                    PixelLabeled = LabeledImage[rowIndex, colIndex]
                    if np.array_equal(PixelLabeled, [0, 0, 0]):
                        result = 0
                    elif np.array_equal(PixelLabeled, [0, 0, 255]):
                        result = 1
                    elif np.array_equal(PixelLabeled, [0, 255, 0]):
                        result = 2
                    elif np.array_equal(PixelLabeled, [255, 0, 0]):
                        result = 3
                    LargerImage[lfY + rowIndex, lfX + colIndex] = result

            # Resize 圖片
            InputImg = cv2.imread(InputFileList[i][j], cv2.IMREAD_GRAYSCALE)
            InputImg = cv2.cvtColor(InputImg, cv2.COLOR_GRAY2BGR)
            InputImg = cv2.resize(InputImg, (480, 360), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(SegNetInputDirList[i] + "/" + str(j + StartIndex) + ".png", InputImg)
            LargerImage = cv2.resize(LargerImage, (480, 360), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(SegNetDirList[i] + "/" + str(j + StartIndex) + ".png", LargerImage)

# 產生 Train.txt
if True:
    TrainTXT = open("train.txt", "w")

    for i in range(len(InputFileList)):
        for j in range(len(InputFileList[i])):
            DataInput = InputFileList[i][j].replace("E:/DentistData/NetworkData", "/home/Dark/SegNet_Data").replace("origin_v2", "SegNet_Input")
            DataLabel = LabeledFileList[i][j].replace("E:/DentistData/NetworkData", "/home/Dark/SegNet_Data").replace("labeled_v2", "SegNet_Labeled")
            DataInput = DataInput.replace("2019.01.08 Tooth", "2019.01.08_Tooth")
            DataLabel = DataLabel.replace("2019.01.08 Tooth", "2019.01.08_Tooth")
            TrainTXT.write(DataInput + " " + DataLabel + "\n")
    TrainTXT.close()