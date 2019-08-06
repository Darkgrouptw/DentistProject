import numpy as np
import cv2
from tqdm import tqdm
import os

DataPath = [
    "E:/DentistData/2019.07.24/data/2019.01.08_ToothBone1",
    # "E:/DentistData/NetworkData/2019.01.08 ToothBone2",
    # "E:/DentistData/2019.07.24/data/2019.01.08_ToothBone3.1",
    # "E:/DentistData/NetworkData/2019.01.08 ToothBone3.2",
    # "E:/DentistData/2019.07.24/data/2019.01.08_ToothBone7.1",
    # "E:/DentistData/2019.07.24/data/2019.01.08_ToothBone8.1",
]
StartIndex = 60
EndIndex = 200

WindowsSize = 101

# 抓取 BoundingBox
def GetBoundingBox(img):
    maxX = 0
    maxY = 0
    minX = 1024
    minY = 1024

    rows, cols = img.shape[:2]
    IsFind = False
    for rowIndex in range(rows):
        for colIndex in range(cols):
            # 拿出此點的
            p = img[rowIndex][colIndex]
            if np.array_equal(p, [255, 0, 0]):
                if rowIndex > maxY:
                    maxY = rowIndex
                elif rowIndex < minY:
                    minY = rowIndex
                elif colIndex > maxX:
                    maxX = colIndex
                elif colIndex < minX:
                    minX = colIndex
                IsFind = True

    if not IsFind:
        minX = 0
        minY = 0
        maxX = 0
        maxY = 0
    return [minX, minY, maxX, maxY]

# 算距離
def Count_IoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])


    # 先算是否有交集
    AreaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    AreaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    InterArea = (xB - xA) * (yB - yA)                       # 交界的面積

    if InterArea < 0 or (InterArea > AreaA and InterArea > AreaB):
        iou = 0
    else:
        iou = InterArea / float(AreaA + AreaB - InterArea)
    print(boxA, boxB, AreaA, AreaB, InterArea, iou)
    return iou

# 跑每一個結果
IoU = []
for i in range(len(DataPath)):
    tempPredictArray = []
    tempLabeledArray = []

    for j in range(StartIndex, EndIndex + 1):
        tempPredict = DataPath[i] + "/Segnetxx/" + str(j) + ".png"            # Predict 位置
        tempLabeled = DataPath[i] + "/Label/" + str(j) + ".png"                # Label 位置

        # IMREAD
        predictImg = cv2.imread(tempPredict)
        labelImg = cv2.imread(tempLabeled)

        # 拿取最大的點
        predict_Bounding = GetBoundingBox(predictImg)
        label_Bounding = GetBoundingBox(labelImg)

        IoU.append(Count_IoU(predict_Bounding, label_Bounding))
IoU = np.array(IoU)
MeanIoU = np.mean(IoU)
# np.append(MeanIoU)
# np.savetxt('IoU.csv', IoU, delimiter='\n')

