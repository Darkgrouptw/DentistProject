import numpy as np
import cv2
from tqdm import tqdm
import os

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

WindowsSize = 101

# 抓取 BoundingBox
def GetBoundingBox(img):
    maxX = 0
    maxY = 0
    minX = 1024
    minY = 1024

    rows, cols = img.shape[:2]
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
    return [minX, minY, maxX, maxY]

# 算距離
def Count_IoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

# 跑每一個結果
IoU = []
for i in range(len(DataPath)):
    tempPredictArray = []
    tempLabeledArray = []

    for j in range(StartIndex, EndIndex + 1):
        tempPredict = DataPath[i] + "/boundingBox_v2/" + str(j) + ".png"            # Predict 位置
        tempLabeled = DataPath[i] + "/labeled_v2/" + str(j) + ".png"                # Label 位置

        # IMREAD
        predictImg = cv2.imread(tempPredict)
        labelImg = cv2.imread(tempLabeled)

        # 拿取最大的點
        predict_Bounding = GetBoundingBox(predictImg)
        label_Bounding = GetBoundingBox(labelImg)

        IoU.append(Count_IoU(predict_Bounding, label_Bounding))
IoU = np.array(IoU)
MeanIoU = np.mean(IoU) / IoU.shape[0]
IoU.append(MeanIoU)
np.savetxt('IoU.csv', IoU, delimiter='\n')

