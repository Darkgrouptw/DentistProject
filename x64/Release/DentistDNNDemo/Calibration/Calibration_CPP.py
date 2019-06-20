import numpy as np
import cv2
import csv
import matlab.engine
import os
# sys.path.append(os.path.dirname(__file__))

# 讀校正檔案
def __ReadCalibrationFile():
    # 讀 csv
    TotalData = []
    script_dir = os.path.dirname(__file__)
    csvPath = os.path.join(script_dir, "./Calibration_CPP.csv")
    with open(csvPath, newline="\n") as csvfile:
        # 抓資料進來
        data = csv.reader(csvfile, delimiter=',')

        # Sum Data
        for rowData in data:
            # 把資料轉成數字
            TempData = []
            for colData in rowData:
                TempData.append(float(colData))

            # 不加空白的東西
            if len(TempData) > 0:
                TotalData.append([TempData])

    TotalData = np.asarray(TotalData, dtype=np.float32)
    print("DataSize: ", TotalData.shape)
    return TotalData

# 分別抓出資料
def __GetData(TotalData):
    rows, _, cols = TotalData.shape
    objP = np.zeros([rows, cols - 2 + 1], np.float32)
    objP[:, :2] = TotalData[:, 0, 2:]

    imgP = TotalData[:, :, :2]
    dataSize, _, channels = imgP.shape
    imgP = imgP.reshape([dataSize, channels])
    imgP = np.array([imgP], dtype=np.float32)
    objP = np.array([objP], dtype=np.float32)

    return imgP, objP
# Main
TotalData = __ReadCalibrationFile()
imgP, objP = __GetData(TotalData)
fig = np.zeros([260, 260], np.float32)
h, w = fig.shape[:2]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objP, imgP, fig.shape[::-1], None, None)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Matlab
eng = matlab.engine.start_matlab()
eng.cd(os.path.dirname(__file__))
TotalDataM = matlab.double(TotalData.tolist())

# API
def CalibrationAPI(dataP):
    # Reshape
    dataP = dataP.reshape([1, -1, 2])

    # UnDistort
    UnDistort = cv2.undistortPoints(dataP, mtx, dist, None, newcameramtx)
    UnDistort = UnDistort.reshape([-1, 2])
    # print(UnDistort.shape)

    # Matlab Data
    UnDistortM = matlab.double(UnDistort.tolist())
    WorldPos = eng.CalibrationTPS(UnDistortM, TotalDataM)
    WorldPos = np.array(WorldPos, dtype=np.float32)
    # print(WorldPos.shape)
    return WorldPos