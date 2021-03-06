import numpy as np
import cv2

class DataManager:
    def __init__(
            self,
            FileNameList,
            LabeledList,
            OutClass,
            WindowsSize,
            # Ratio = 0.9
    ):
        # 前置判斷
        print("DataSize: ", len(FileNameList))
        print("LabeledSize: ", len(LabeledList))
        assert(len(FileNameList) == len(LabeledList))
        assert WindowsSize % 2 == 1, "目前先測試 Mod 有餘數的部分"
        self.WindowsSize = WindowsSize
        self.OutClass = OutClass

        # 讀檔案
        self._ReadData(FileNameList, LabeledList)
        print("Package Data Size:", self.DataSize)
        print("Rotate Package Data Size:", self.DataSize)

        # 參數設定
        # self.TrainValidRatio = Ratio



    # 拿一部分的 Train Data
    def BatchTrainData(self, size):
        # choice = np.random.randint(int(self.DataSize * self.TrainValidRatio), size=size)
        AQuarterSize = int(size / 4)
        choiceIndexNoneZero = np.random.choice(self.NoneZeroIndexArray, size=AQuarterSize, replace=False)
        choiceIndexZero = np.random.choice(self.ZerosIndexArray, size=AQuarterSize, replace=False)
        choiceIndex = np.concatenate([choiceIndexNoneZero, choiceIndexZero], axis=0)
        choiceData = self.Data[choiceIndex]
        choiceLabelData = self.LabelData[choiceIndex]
        # print(choiceLabelData.shape)

        choiceIndexRotateNoneZero = np.random.choice(self.NoneZeroRotateIndexArray, size=AQuarterSize, replace=False)
        choiceIndexRotateZero = np.random.choice(self.ZerosRotateIndexArray, size=AQuarterSize, replace=False)
        choiceRotateIndex = np.concatenate([choiceIndexRotateNoneZero, choiceIndexRotateZero], axis=0)
        choiceRotateData = self.RotateData[choiceRotateIndex]
        choiceRotateLabelData = self.RotateLabelData[choiceRotateIndex]
        # print(choiceRotateLabelData.shape)

        TotalData = np.concatenate([choiceData, choiceRotateData], axis=0)
        TotalLabelData = np.concatenate([choiceLabelData, choiceRotateLabelData], axis=0)
        # print(TotalData.shape)
        # print(TotalLabelData.shape)
        # assert False
        return TotalData.reshape(size, self.WindowsSize, self.WindowsSize, 1), TotalLabelData.reshape(size, self.OutClass)

    # 拿 Valid Data
    def BatchValidData(self, size):
        choice = np.random.randint(int(self.DataSize * (1 - self.TrainValidRatio)), size=size) + int(self.DataSize * self.TrainValidRatio)
        return self.Data[choice].reshape(size, self.WindowsSize, self.WindowsSize, 1), self.LabelData[choice].reshape(size, self.OutClass)

    # 測試一張圖
    def TestFirstNBoxOfTrainData(self, size):
        return self.Data[: self.rows * self.cols * size].reshape(self.rows * self.cols * size, self.WindowsSize, self.WindowsSize, 1)
        # return self.Data[:size].reshape(size, self.WindowsSize, self.WindowsSize, 1)

    def TestFirstNBoxOfValidData(self, size):
        # offset = int(self.DataSize * self.TrainValidRatio)
        # return self.Data[offset:offset + size].reshape(size, self.Rows, self.Cols, 1)
        pass


    #
    # Helper Function
    #

    # 把資料載進來
    def _ReadData(self, FileNameList, LabeledList):
        # 算長度 & 初始化資料大小的 Array
        self.rows, self.cols = cv2.imread(FileNameList[0], cv2.IMREAD_GRAYSCALE).shape
        self.DataSize = len(FileNameList) * self.rows * self.cols       # Rotate

        # 由於資料不平均
        # 所以這邊有作一些修正
        self.NoneZeroIndexArray = []
        self.ZerosIndexArray = []
        self.NoneZeroRotateIndexArray = []
        self.ZerosRotateIndexArray = []

        # 讀 Input
        self.Data = np.zeros([self.DataSize, self.WindowsSize, self.WindowsSize], np.float32)
        self.RotateData = np.zeros([self.DataSize, self.WindowsSize, self.WindowsSize], np.float32)
        self.LabelData = np.zeros([self.DataSize, self.OutClass], np.float32)
        self.RotateLabelData = np.zeros([self.DataSize, self.OutClass], np.float32)
        for i in range(len(FileNameList)):
            # 讀圖
            InputImg = cv2.imread(FileNameList[i], cv2.IMREAD_GRAYSCALE) / 255
            LabelImg = cv2.imread(LabeledList[i], cv2.IMREAD_COLOR)

            LabelProbImg = self._GetProbBorderImg(LabelImg)

            # 先產生大張的圖
            halfRadius = int((self.WindowsSize - 1) / 2)
            LargerInputImg = np.zeros([halfRadius * 2 + InputImg.shape[0], halfRadius * 2 + InputImg.shape[1]], np.float32)
            LargerInputImg[halfRadius:halfRadius + InputImg.shape[0], halfRadius:halfRadius + InputImg.shape[1]] = InputImg

            # Rotate
            LargerInputImgRotate = self._RotateImage(LargerInputImg, 180)
            LabelProbImgRotate = self._RotateImage(LabelProbImg, 180)
            for rowIndex in range(0, self.rows):
                for colIndex in range(0, self.cols):
                    # 塞進值
                    InputDataTemp = LargerInputImg[rowIndex: rowIndex + self.WindowsSize, colIndex: colIndex + self.WindowsSize]
                    DataIndex = i * self.rows * self.cols + rowIndex * self.cols + colIndex
                    self.Data[DataIndex] = InputDataTemp

                    Prob = LabelProbImg[rowIndex][colIndex]
                    self.LabelData[DataIndex] = Prob

                    # 這邊要多增加 Index 加入
                    if Prob != 0:
                        self.NoneZeroIndexArray.append(DataIndex)
                    else:
                        self.ZerosIndexArray.append(DataIndex)

                    # RotatePart
                    InputRotateDataTemp = LargerInputImgRotate[rowIndex: rowIndex + self.WindowsSize, colIndex: colIndex + self.WindowsSize]
                    self.RotateData[DataIndex] = InputRotateDataTemp

                    ProbRotate = LabelProbImgRotate[rowIndex][colIndex]
                    self.RotateLabelData[DataIndex] = ProbRotate

                    # 這邊要多增加 Index 加入
                    if ProbRotate != 0:
                        self.NoneZeroRotateIndexArray.append(DataIndex)
                    else:
                        self.ZerosRotateIndexArray.append(DataIndex)

        # 轉成 numpy
        self.ZerosIndexArray = np.asarray(self.ZerosIndexArray)
        self.NoneZeroIndexArray = np.asarray(self.NoneZeroIndexArray)
        self.ZerosRotateIndexArray = np.asarray(self.ZerosRotateIndexArray)
        self.NoneZeroRotateIndexArray = np.asarray(self.NoneZeroRotateIndexArray)
        print("None Zero Shape: ", self.NoneZeroIndexArray.shape)
        print("Zero Shape: ", self.ZerosIndexArray.shape)
        print("Rotate None Zero Shape: ", self.NoneZeroRotateIndexArray.shape)
        print("Rotate Zero Shape: ", self.ZerosRotateIndexArray.shape)

    # 旋轉 180 度
    def _RotateImage(self, img, angle):
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotatedImg = cv2.warpAffine(img, M, (h, w))
        return rotatedImg


    # 把圖片轉換為機率圖片
    def _GetProbBorderImg(self, LabelImg):
        # 常態分佈表
        GaussianDis = [0.12074, 0.28961, 0.40842, 0.54106, 0.67336, 0.78724, 0.86462]

        # 要回傳的圖片
        LabelProbImg = np.zeros(LabelImg.shape[0:2], np.float32)

        # Row 到每個 Col
        for rowIndex in range(LabelImg.shape[0]):
            # 是否中間有斷點
            IsMeetBlue = False
            IsMeetRed = False
            BreakCol = -1
            for colIndex in range(LabelImg.shape[1]):
                color = LabelImg[rowIndex][colIndex]
                # print(colIndex)
                if np.array_equal(color, [255, 0, 0]):
                    IsMeetBlue = True
                if np.array_equal(color, [0, 0, 255]):
                    IsMeetRed = True

                if IsMeetRed and IsMeetBlue:
                    BreakCol = colIndex
                    break

            # 有找到分界線
            if BreakCol != -1:
                LabelProbImg[rowIndex][BreakCol] = 1
                # 往上
                for colIndex in range(len(GaussianDis)):
                    colTemp = BreakCol - 1 - colIndex
                    if colTemp >= 0:
                        LabelProbImg[rowIndex][colTemp] += GaussianDis[len(GaussianDis) - 1 - colIndex]
                    else:
                        break

                # 往下
                for colIndex in range(len(GaussianDis)):
                    colTemp = BreakCol + 1 + colIndex
                    if colTemp < LabelImg.shape[1]:
                        LabelProbImg[rowIndex][colTemp] += GaussianDis[len(GaussianDis) - 1 - colIndex]
                    else:
                        break

        # Col 到每個 Row
        for colIndex in range(LabelImg.shape[1]):
            # 是否中間有斷點
            IsMeetBlue = False
            IsMeetRed = False
            BreakRow = -1
            for rowIndex in range(LabelImg.shape[0]):
                color = LabelImg[rowIndex][colIndex]

                if np.array_equal(color, [255, 0, 0]):
                    IsMeetBlue = True
                if np.array_equal(color, [0, 0, 255]):
                    IsMeetRed = True

                if IsMeetRed and IsMeetBlue:
                    BreakRow = rowIndex
                    break

            # 有找到分界線
            if BreakRow != -1:
                LabelProbImg[BreakRow][colIndex] = 1
                # 往上
                for rowIndex in range(len(GaussianDis)):
                    rowTemp = BreakRow - 1 - rowIndex
                    if rowTemp >= 0:
                        LabelProbImg[rowTemp][colIndex] = GaussianDis[len(GaussianDis) - 1 - rowIndex] if LabelProbImg[rowTemp][colIndex] < GaussianDis[len(GaussianDis) - 1 - rowIndex] else LabelProbImg[rowTemp][colIndex]
                    else:
                        break

                # 往下
                for rowIndex in range(len(GaussianDis)):
                    rowTemp = BreakRow + 1 + rowIndex
                    if rowTemp < LabelImg.shape[0]:
                        LabelProbImg[rowTemp][colIndex] = GaussianDis[len(GaussianDis) - 1 - rowIndex] if LabelProbImg[rowTemp][colIndex] < GaussianDis[len(GaussianDis) - 1 - rowIndex] else LabelProbImg[rowTemp][colIndex]
                    else:
                        break
        LabelProbImg = np.clip(LabelProbImg, 0, 1)
        return LabelProbImg