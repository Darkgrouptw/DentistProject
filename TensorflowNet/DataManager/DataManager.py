import numpy as np
import cv2
from tqdm import tqdm
import os

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
        print("DataSize (set): ", len(FileNameList))
        print("LabeledSize (set): ", len(LabeledList))
        assert(len(FileNameList) == len(LabeledList))
        assert WindowsSize % 2 == 1, "目前先測試 Mod 有餘數的部分"
        self.WindowsSize = WindowsSize
        self.OutClass = OutClass

        # 讀檔案
        self._ReadData(FileNameList, LabeledList)

        # 參數設定
        # self.TrainValidRatio = Ratio

    # 拿一部分的 Train Data
    def BatchTrainData(self, size):
        # 抓出所有要的 Index 來產生資料
        halfDataSize = int(size / 2)
        choiceIndexNonZero = self.NonZeroIndexArray[np.random.choice(self.NonZeroIndexArray.shape[0], size=halfDataSize, replace=False)]
        choiceIndexZero = self.ZeroIndexArray[np.random.choice(self.ZeroIndexArray.shape[0], size=halfDataSize, replace=False)]
        TotalIndex = np.concatenate([choiceIndexNonZero, choiceIndexZero], axis=0)
        return self._GetWindowImgFromPos(TotalIndex)


    # 拿 Valid Data
    def BatchValidData(self, size):
        # choice = np.random.randint(int(self.DataSize * (1 - self.TrainValidRatio)), size=size) + int(self.DataSize * self.TrainValidRatio)
        # return self.Data[choice].reshape(size, self.WindowsSize, self.WindowsSize, 1), self.LabelData[choice].reshape(size, self.OutClass)
        pass

    # 測試一組圖
    # def TestFirstNBoxOfTrainData(self, size):
    #     # 只取前面幾個
    #     totalSize = size * (200 - 60 + 1)
    #     _, endIndex = self.StartEndIndex[totalSize - 1]
    #
    #     return self._GetWindowImgFromPos(self.IndexArray[0: endIndex])
    #
    # def TestFirstNBoxOfValidData(self, size):
    #     pass

    # 測試一張圖
    def TestFullImage(self, index):
        startIndex, endIndex, rows, cols = self.StartEndIndex[index]
        InputData, _ = self._GetWindowImgFromPos(self.IndexArray[startIndex:endIndex])
        return InputData, rows, cols


    #
    # Helper Function
    #

    # 把整張圖讀進來
    def _ReadData(self, FileNameList, LabeledList):
        print("Start Read Data!")
        assert len(FileNameList) == len(LabeledList), "DataSize is not the same."

        # 由於資料不平均
        # 所以這邊有作一些修正
        # 存放的內容 [rowIndex, colIndex]
        self.NonZeroIndexArray = []
        self.ZeroIndexArray = []
        self.IndexArray = []

        # 建造參數
        TotalSize = len(FileNameList) * (200 - 60 + 1)
        self.Data = []                                                      # 由於圖片大小不確定，所以只能給這種
        self.LabelData = []                                                 # 同上
        self.StartEndIndex = np.zeros([TotalSize, 4], np.int32)             # [StartIndex, EndIndex, rows, cols]，[Start, EndIndex)，用在 Test 前面幾張用的

        DataPixelCount = 0
        for i in range(len(FileNameList)):          # 跳過最後一個 \n
            # Debug 用
            print("Read DataSet: ", i, "/", str(len(FileNameList)))

            for j in tqdm(range(len(FileNameList[i]))):
                # 讀圖，並確保有讀到
                InputImg = cv2.imread(FileNameList[i][j], cv2.IMREAD_GRAYSCALE) / 255
                LabelImg = cv2.imread(LabeledList[i][j])

                # 確保有讀到資料
                assert type(InputImg) is np.ndarray and type(LabelImg) is np.ndarray
                rows, cols = InputImg.shape[:2]
                LabelProbImg = self._TransformProbImg(LabelImg)

                # 抓出所有零 與 非零 的
                for rowIndex in range(rows):
                    for colIndex in range(cols):
                        Prob = LabelProbImg[rowIndex][colIndex]
                        PosInfo = [len(self.Data), rowIndex, colIndex]
                        if np.array_equal(Prob, [1, 0, 0, 0]):
                            self.NonZeroIndexArray.append(PosInfo)
                        else:
                            self.ZeroIndexArray.append(PosInfo)
                        self.IndexArray.append(PosInfo)

                # 計算哪一張圖對應的
                startIndex = DataPixelCount
                DataPixelCount += rows * cols
                endIndex = DataPixelCount
                self.StartEndIndex[len(self.Data)] = [startIndex, endIndex, rows, cols]

                # 圖片結算
                self.Data.append(InputImg)
                self.LabelData.append(LabelProbImg)

        # 讀完檔案
        print("Finish Read Data!")

        # 最後再轉乘 Numpy Array
        # self.Data = np.asarray(self.Data)
        # self.LabelData = np.asarray(self.LabelData)
        self.ZeroIndexArray = np.asarray(self.ZeroIndexArray)
        self.NonZeroIndexArray = np.asarray(self.NonZeroIndexArray)
        self.IndexArray = np.asarray(self.IndexArray)

        print("====================================================")
        print("DataSize: ", len(self.Data))
        print("Label: ", len(self.LabelData))
        print("IndexArray: ", self.IndexArray.shape)
        print("ZeroIndex: ", self.ZeroIndexArray.shape)
        print("NonZeroIndex: ", self.NonZeroIndexArray.shape)
        print("====================================================")

    # 把圖片轉換 ArgMax 的 Class Array
    def _TransformProbImg(self, LabelImg):
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
                    assert False, "測試有沒有 Bug"
        return LabelProbImg

    def _GetWindowImgFromPos(self, choosePos):
        size = choosePos.shape[0]
        WindowData = np.zeros([size, self.WindowsSize, self.WindowsSize], np.float32)
        WindowLabel = np.zeros([size, self.OutClass], np.float32)

        # 接這要跑每一筆資料加到陣列中
        for i in range(size):
            # 抓出位置
            pos = choosePos[i]

            # 抓取原圖
            InputImg = self.Data[pos[0]]
            LabelProbImg = self.LabelData[pos[0]]

            # 放大
            halfRadius = int((self.WindowsSize - 1) / 2)
            LargerInputImg = np.zeros([halfRadius * 2 + InputImg.shape[0], halfRadius * 2 + InputImg.shape[1]], np.float32)
            LargerInputImg[halfRadius:halfRadius + InputImg.shape[0], halfRadius:halfRadius + InputImg.shape[1]] = InputImg

            # 設定原本要放的位置
            rowIndex = pos[1]
            colIndex = pos[2]

            WindowData[i] = LargerInputImg[rowIndex: rowIndex + self.WindowsSize, colIndex: colIndex + self.WindowsSize]
            WindowLabel[i] = LabelProbImg[rowIndex][colIndex]
        return WindowData.reshape([size, self.WindowsSize, self.WindowsSize, 1]), WindowLabel