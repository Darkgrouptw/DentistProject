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

        # 參數設定
        # self.TrainValidRatio = Ratio



    # 拿一部分的 Train Data
    def BatchTrainData(self, size):
        # choice = np.random.randint(int(self.DataSize * self.TrainValidRatio), size=size)
        halfSize = int(size / 2)
        choiceNoneZero = np.random.choice(self.NoneZeroIndexArray, size=halfSize, replace=False)
        choiceZero = np.random.choice(self.ZerosIndexArray, size=halfSize, replace=False)
        choice = np.concatenate([choiceNoneZero, choiceZero], axis=0)
        return self.Data[choice].reshape(size, self.WindowsSize, self.WindowsSize, 1), self.LabelData[choice].reshape(size, self.OutClass)

    # 拿 Valid Data
    def BatchValidData(self, size):
        choice = np.random.randint(int(self.DataSize * (1 - self.TrainValidRatio)), size=size) + int(self.DataSize * self.TrainValidRatio)
        return self.Data[choice].reshape(size, self.WindowsSize, self.WindowsSize, 1), self.LabelData[choice].reshape(size, self.OutClass)

    # 測試一張圖
    def TestFirstNBoxOfTrainData(self, size):
        return self.Data[:size].reshape(size, self.WindowsSize, self.WindowsSize, 1)

    def TestFirstNBoxOfValidData(self, size):
        offset = int(self.DataSize * self.TrainValidRatio)
        return self.Data[offset:offset + size].reshape(size, self.Rows, self.Cols, 1)


    #
    # Helper Function
    #

    # 把資料載進來
    def _ReadData(self, FileNameList, LabeledList):
        # 算長度 & 初始化資料大小的 Array
        rows, cols = cv2.imread(FileNameList[0], cv2.IMREAD_GRAYSCALE).shape
        self.DataSize = len(FileNameList) * rows * cols

        # 由於資料不平均
        # 所以這邊有作一些修正
        self.NoneZeroIndexArray = []
        self.ZerosIndexArray = []

        # 讀 Input
        self.Data = np.zeros([self.DataSize, self.WindowsSize, self.WindowsSize], np.float32)
        self.LabelData = np.zeros([self.DataSize, self.OutClass], np.float32)
        for i in range(len(FileNameList)):
            # 讀圖
            InputImg = cv2.imread(FileNameList[i], cv2.IMREAD_GRAYSCALE)
            LabelImg = cv2.imread(LabeledList[i], cv2.IMREAD_COLOR)

            LabelProbImg = self._GetProbBorderImg(LabelImg)

            # 先產生大張的圖
            halfRadius = int((self.WindowsSize - 1) / 2)
            LargerInputImg = np.zeros([halfRadius * 2 + InputImg.shape[0], halfRadius * 2 + InputImg.shape[1]], np.float32)
            LargerInputImg[halfRadius:halfRadius + InputImg.shape[0], halfRadius:halfRadius + InputImg.shape[1]] = InputImg
            for rowIndex in range(0, rows):
                for colIndex in range(0, cols):
                    # 塞進值
                    InputDataTemp = LargerInputImg[rowIndex: rowIndex + self.WindowsSize, colIndex: colIndex + self.WindowsSize]
                    DataIndex = i * rows * cols + rowIndex * cols + colIndex

                    self.Data[DataIndex] = InputDataTemp

                    Prob = LabelProbImg[rowIndex][colIndex]
                    self.LabelData[DataIndex] = Prob

                    # 這邊要多增加 Index 加入
                    if Prob != 0:
                        self.NoneZeroIndexArray.append(DataIndex)
                    else:
                        self.ZerosIndexArray.append(DataIndex)

        # 轉成 numpy
        self.ZerosIndexArray = np.asarray(self.ZerosIndexArray)
        self.NoneZeroIndexArray = np.asarray(self.NoneZeroIndexArray)
        print("None Zero Shape: ", self.NoneZeroIndexArray.shape)
        print("Zero Shape: ", self.ZerosIndexArray.shape)

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