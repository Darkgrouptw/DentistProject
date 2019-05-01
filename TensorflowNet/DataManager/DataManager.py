import numpy as np
import cv2
from tqdm import tqdm

# Label
# 0 背景
# 1 牙齒
# 2 牙齦
# 3 齒槽骨

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

        # Index 設定
        # x由於這個資料及大小會有大有小，所以必須要有一個來存一張開始翰結束
        # x[ StartIndex[0], EndIndexArray[0] )
        self.IndexArray = []
        self.RowIndexArray = []
        self.ColIndexArray = []
        self.Data = []
        self.LabelData = []

        # 讀檔案
        self._ReadData(FileNameList, LabeledList)
        print("Package Data Size:", self.DataSize)
        print("Rotate Package Data Size:", self.DataSize)

        # 參數設定
        # self.TrainValidRatio = Ratio



    # 拿一部分的 Train Data
    def BatchTrainData(self, size):
        # choice = np.random.randint(int(self.DataSize * self.TrainValidRatio), size=size)
        # 抓出所有要的 Index 來產生資料
        halfDataSize = int(size / 2)
        choiceIndexNoneZero = np.random.choice(self.NoneZeroIndexArray, size=halfDataSize, replace=False)
        choiceIndexZero = np.random.choice(self.ZerosIndexArray, size=halfDataSize, replace=False)
        choiceIndex = np.concatenate([choiceIndexNoneZero, choiceIndexZero], axis=0)
        return self._GetWindowImgFromIndex(choiceIndex)
        # choiceData = self.Data[choiceIndex]
        # choiceLabelData = self.LabelData[choiceIndex]
        # print(choiceLabelData.shape)

        # TotalData = choiceData
        # TotalLabelData = choiceLabelData
        # print(TotalData.shape)
        # print(TotalLabelData.shape)TestFirstNBoxOfTrainData
        # assert False
        # return TotalData.reshape(size, self.WindowsSize, self.WindowsSize, 1), TotalLabelData.reshape(size, self.OutClass)

    # 拿 Valid Data
    def BatchValidData(self, size):
        # choice = np.random.randint(int(self.DataSize * (1 - self.TrainValidRatio)), size=size) + int(self.DataSize * self.TrainValidRatio)
        # return self.Data[choice].reshape(size, self.WindowsSize, self.WindowsSize, 1), self.LabelData[choice].reshape(size, self.OutClass)
        pass

    # 測試一張圖
    def TestFirstNBoxOfTrainData(self, size):
        startIndex = self.StartIndexArray[0]
        endIndex = self.EndIndexArray[size * (200 - 60 + 1) - 1]
        # print(self.Data.shape)
        # print(startIndex, endIndex)
        return self.Data[startIndex: endIndex].reshape(-1, self.WindowsSize, self.WindowsSize, 1)
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
        # 由於資料不平均
        # 所以這邊有作一些修正
        self.NoneZeroIndexArray = []
        self.ZerosIndexArray = []

        # 算長度 & 初始化資料大小的 Array
        self.DataSize = 0
        DataIndex = 0
        print("Start Reading File!!")

        for i in tqdm(range(len(FileNameList))):
            # 讀圖
            InputImg = cv2.imread(FileNameList[i], cv2.IMREAD_GRAYSCALE) / 255  # 這邊要除 255
            LabelImg = cv2.imread(LabeledList[i], cv2.IMREAD_COLOR)
            LabelProbImg = self._TransformProbImg(LabelImg, i)

            # 加到陣列裡
            self.Data.append(InputImg)
            self.LabelData.append(LabelProbImg)
            rows, cols = InputImg.shape[:2]

            # 找出哪邊是 0 & 非零
            for rowIndex in range(rows):
                for colIndex in range(cols):
                    # 產生方便 Tracking 的陣列
                    self.IndexArray.append(i)
                    self.RowIndexArray.append(rowIndex)
                    self.ColIndexArray.append(colIndex)

                    Prob = LabelProbImg[rowIndex][colIndex]

                    if np.array_equal(Prob, [1, 0, 0, 0]):
                        self.NoneZeroIndexArray.append(DataIndex)
                    else:
                        self.ZerosIndexArray.append(DataIndex)
                    DataIndex += 1

        # 算位置 (正反)
        self.DataSize = DataIndex       # 理論上這個變數可以省略，但來不及啦!!!
        print("Read File Done!!")


        # for i in tqdm(range(len(FileNameList))):
        #     # 讀圖
        #     InputImg = cv2.imread(FileNameList[i], cv2.IMREAD_GRAYSCALE) / 255          # 這邊要除 255
        #     LabelImg = cv2.imread(LabeledList[i], cv2.IMREAD_COLOR)
        #
        #     # LabelProbImg = self._GetProbBorderImg(LabelImg)
        #     rows, cols = InputImg.shape[:2]
        #     LabelProbImg = self._TransformProbImg(LabelImg, i)
        #
        #     # 先產生大張的圖
        #     halfRadius = int((self.WindowsSize - 1) / 2)
        #     LargerInputImg = np.zeros([halfRadius * 2 + InputImg.shape[0], halfRadius * 2 + InputImg.shape[1]], np.float32)
        #     LargerInputImg[halfRadius:halfRadius + InputImg.shape[0], halfRadius:halfRadius + InputImg.shape[1]] = InputImg
        #     # print("A", LargerInputImg.shape)
        #
        #     # 加入 Index
        #     self.StartIndexArray.append(DataIndex)
        #     self.EndIndexArray.append(DataIndex + rows * cols)
        #     self.RowArray.append(rows)
        #     self.ColArray.append(cols)
        #
        #     for rowIndex in range(0, rows):
        #         for colIndex in range(0, cols):
        #             # 塞進值
        #             # print(rowIndex, rowIndex + self.WindowsSize, colIndex, colIndex + self.WindowsSize)
        #             InputDataTemp = LargerInputImg[rowIndex: rowIndex + self.WindowsSize, colIndex: colIndex + self.WindowsSize]
        #             self.Data[DataIndex] = InputDataTemp
        #
        #             Prob = LabelProbImg[rowIndex][colIndex]
        #             self.LabelData[DataIndex] = Prob
        #
        #             # 這邊要多增加 Index 加入
        #             if np.array_equal(Prob, [1, 0, 0, 0]):
        #                 self.NoneZeroIndexArray.append(DataIndex)
        #             else:
        #                 self.ZerosIndexArray.append(DataIndex)
        #             DataIndex += 1
        #
        # # 轉成 numpy
        # self.ZerosIndexArray = np.asarray(self.ZerosIndexArray)
        # self.NoneZeroIndexArray = np.asarray(self.NoneZeroIndexArray)
        # print("None Zero Shape: ", self.NoneZeroIndexArray.shape)
        # print("Zero Shape: ", self.ZerosIndexArray.shape)

    # 旋轉 180 度
    def _RotateImage(self, img, angle):
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotatedImg = cv2.warpAffine(img, M, (h, w))
        return rotatedImg

    # 抓出 Border 再把它轉成圖片
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

    # 把圖片轉換 ArgMax 的 Class Array
    def _TransformProbImg(self, LabelImg, i):
        # LabelProbImg = np.zeros([LabelImg.shape[0:2]])
        shapeSize = np.concatenate([LabelImg.shape[0:2], [self.OutClass]], axis=0)
        LabelProbImg = np.zeros(shapeSize, np.float32)

        # Row 到每個 Col
        for rowIndex in range(LabelImg.shape[0]):
            for colIndex in range(LabelImg.shape[1]):
                # 背景
                p = LabelImg[rowIndex][colIndex]
                if np.array_equal(p, [0, 0, 0]):
                    LabelProbImg[rowIndex][colIndex][0] = 1
                elif np.array_equal(p, [0, 0, 255]):
                    LabelProbImg[rowIndex][colIndex][1] = 1
                elif np.array_equal(p, [0, 255, 0]):
                    LabelProbImg[rowIndex][colIndex][2] = 1
                elif np.array_equal(p, [255, 0, 0]):
                    LabelProbImg[rowIndex][colIndex][3] = 1
                else:
                    print(i)
                    assert False, "測試有沒有 Bug"
        return LabelProbImg

    # 根據這些 Index 來抓出圖片的範圍
    def _GetWindowImgFromIndex(self, chooseIndex):
        size = chooseIndex.shape[0]
        WindowData = np.zeros([size, self.WindowsSize, self.WindowsSize], np.float32)
        WindowLabel = np.zeros([size, self.OutClass], np.float32)

        # 接這跑每一個筆資料專出位置之後並且加進陣列中
        for i in range(size):
            # 先產生大張的圖
            halfRadius = int((self.WindowsSize - 1) / 2)

            # 抓取原圖
            InputImg = self.Data[self.IndexArray[chooseIndex[i]]]
            LabelProbImg = self.LabelData[self.IndexArray[chooseIndex[i]]]
            LargerInputImg = np.zeros([halfRadius * 2 + InputImg.shape[0], halfRadius * 2 + InputImg.shape[1]], np.float32)
            LargerInputImg[halfRadius:halfRadius + InputImg.shape[0], halfRadius:halfRadius + InputImg.shape[1]] = InputImg

            # 放進原本該放的地方
            rowIndex = self.RowIndexArray[chooseIndex[i]]
            colIndex = self.ColIndexArray[chooseIndex[i]]

            WindowData[i] = LargerInputImg[rowIndex: rowIndex + self.WindowsSize, colIndex: colIndex + self.WindowsSize]
            WindowLabel[i] = LabelProbImg[rowIndex][colIndex]
        return WindowData.reshape([size, self.WindowsSize, self.WindowsSize, 1]), WindowLabel.reshape([size, self.OutClass])