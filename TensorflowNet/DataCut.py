from PrePrecess.PointClass import Point
import numpy as np
import cv2
import os
from tqdm import tqdm

# 路徑設定
DoListPath = [
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
]

# 參數設定
StartIndex = 60
EndIndex = 200


# 先加入要做的東西到 DoList 到
# DoListArray = []
ImgArray = []
for i in tqdm(range(len(DoListPath))):
    # DoListArray.append()
    boundingBoxFile = os.path.join(DoListPath[i], "./boundingBox.txt")
    boundingF = open(boundingBoxFile, "r")
    boundingContent = boundingF.read()
    boundingLines = boundingContent.split("\n")


    # 加資料夾
    ImgTempDir = os.path.join(DoListPath[i], "./boundingBox_v2")
    if not os.path.exists(ImgTempDir):
        os.makedirs(ImgTempDir)

    ImgTemp = []
    for j in range(len(boundingLines)):
        if j - 1 >= StartIndex and j - 1 <= EndIndex:
            numTemp = boundingLines[j].split(" ")

            # 先判斷檔案正確性
            assert(len(numTemp) >= 4)
            tl = Point(int(numTemp[0]), int(numTemp[1]))
            br = Point(int(numTemp[2]), int(numTemp[3]))

            ImgPath = os.path.join(DoListPath[i], "./origin_v2/" + str(j - 1) + ".png")
            img = cv2.imread(ImgPath)
            img = img[tl.Y:br.Y, tl.X:br.X]

            # Img 加起來
            boundingPath = os.path.join(DoListPath[i], "./boundingBox_v2/" + str(j - 1) + ".png")
            ImgTemp.append(img)
            cv2.imwrite(boundingPath, img)
    # Img Array
    ImgArray.append(ImgTemp)
    boundingF.close()
