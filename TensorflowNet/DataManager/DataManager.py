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

        #檔案前處理
        self.DataListDir = "./DataList"
        if not self._CheckFileListIsExist():
            print("Creating Data!!")
            self._ReadDataAndCreateListData(FileNameList, LabeledList)
        else:
            print("Already has Data!!")

        # 讀檔案
        self.ImgPath = []                   # 圖片位置
        self.LabelResult = []               # 存非 0 的結果
        self.NonZeroIndexArray = []         # 存非 0 的方哪
        self.ZeroIndexArray = []            # 存 0 放在哪
        self._ReadListData()

        # 參數設定
        # self.TrainValidRatio = Ratio



    # 拿一部分的 Train Data
    def BatchTrainData(self, size):
        # 抓出所有要的 Index 來產生資料
        halfDataSize = int(size / 2)
        choiceIndexNonZero = np.random.choice(self.NonZeroIndexArray, size=halfDataSize, replace=False)
        choiceIndexZero = np.random.choice(self.ZeroIndexArray, size=halfDataSize, replace=False)
        TotalData = np.zeros([size, self.WindowsSize, self.WindowsSize])
        TotalLabelData = np.zeros([size, self.OutClass])

        # 抓圖片
        for i in range(halfDataSize):
            # 前半對是 Zero
            index = choiceIndexZero[i]
            TotalData[i] = cv2.imread(self.ImgPath[index], cv2.IMREAD_GRAYSCALE) / 255
            TotalLabelData[i] = self.LabelResult[index]

            # 後半段是 NonZero
            index = choiceIndexNonZero[i]
            TotalData[halfDataSize + i] = cv2.imread(self.ImgPath[index], cv2.IMREAD_GRAYSCALE) / 255
            TotalLabelData[halfDataSize + i] = self.LabelResult[index]
        return TotalData.reshape([size, self.WindowsSize, self.WindowsSize, 1]), TotalLabelData.reshape([size, self.OutClass])

    # 拿 Valid Data
    def BatchValidData(self, size):
        # choice = np.random.randint(int(self.DataSize * (1 - self.TrainValidRatio)), size=size) + int(self.DataSize * self.TrainValidRatio)
        # return self.Data[choice].reshape(size, self.WindowsSize, self.WindowsSize, 1), self.LabelData[choice].reshape(size, self.OutClass)
        pass

    # 測試一張圖
    def TestFirstNBoxOfTrainData(self, size):
        pass

    def TestFirstNBoxOfValidData(self, size):
        # offset = int(self.DataSize * self.TrainValidRatio)
        # return self.Data[offset:offset + size].reshape(size, self.Rows, self.Cols, 1)
        pass


    #
    # Helper Function
    #

    # 先確定檔案室否有建立起來
    def _CheckFileListIsExist(self):
        # 使否有資料夾
        # 這便主要有三件事情
        # 1. 見檔案資料夾
        # 2. 建 Data List
        if not os.path.isdir(self.DataListDir):
            os.mkdir(self.DataListDir)
            return False

        if not os.path.exists(self.DataListDir + "/DataList.txt"):
            return False
        return True

    # 把資料載進來，然後到目錄中產生 兩個 List 檔案
    def _ReadDataAndCreateListData(self, FileNameList, LabeledList):
        # 判斷資料夾存不存在
        if not os.path.isdir(self.DataListDir + "/Data"):
            os.mkdir(self.DataListDir + "/Data")

        # 算長度 & 初始化資料大小的 Array
        print("Start to Generate Data!!")

        # 打開 Total 的檔案
        DataList = open(self.DataListDir + "/DataList.txt", "w")

        for i in range(len(FileNameList)):
            print("Creating DataSet: ",i, "/", str(len(FileNameList)))

            for j in tqdm(range(len(FileNameList[i]))):
                # 先找出要存檔在哪裡
                AllIndex = [index for index in range(len(FileNameList[i][j])) if FileNameList[i][j].startswith('/', index)]
                PNGIndex = FileNameList[i][j].rfind(".png")

                # 如果是第 0，就要創建資料夾
                if j == 0:
                    # 創建並寫入 Datalist 中
                    DataListSinglePath = self.DataListDir + "/Data/" + FileNameList[i][j][AllIndex[-3] + 1: AllIndex[-2]] + ".txt"
                    DataListSingle = open(DataListSinglePath, "w")
                    DataList.write(DataListSinglePath + "\n")

                    # 產生 Data
                    os.mkdir(self.DataListDir + "/Data/" + FileNameList[i][j][AllIndex[-3] + 1: AllIndex[-2]])

                SaveDataName = self.DataListDir + "/Data/" + FileNameList[i][j][AllIndex[-3] + 1: AllIndex[-2]] + "/" + FileNameList[i][j][AllIndex[-1] + 1:PNGIndex]

                # 讀圖
                InputImg = cv2.imread(FileNameList[i][j], cv2.IMREAD_GRAYSCALE) / 255  # 這邊要除 255
                LabelImg = cv2.imread(LabeledList[i][j], cv2.IMREAD_COLOR)
                LabelProbImg = self._TransformProbImg(LabelImg, i)

                # 找出哪邊是 0 & 非零
                [rows, cols] = InputImg.shape[:2]
                for rowIndex in range(rows):
                    for colIndex in range(cols):
                        # 抓取原圖
                        halfRadius = int((self.WindowsSize - 1) / 2)

                        # 抓取原圖
                        LargerInputImg = np.zeros([halfRadius * 2 + InputImg.shape[0], halfRadius * 2 + InputImg.shape[1]], np.float32)
                        LargerInputImg[halfRadius:halfRadius + InputImg.shape[0], halfRadius:halfRadius + InputImg.shape[1]] = InputImg

                        # 放進原本該放的地方
                        WindowImg = LargerInputImg[rowIndex: rowIndex + self.WindowsSize,
                                        colIndex: colIndex + self.WindowsSize]
                        WindowProb = LabelProbImg[rowIndex][colIndex]
                        CurrentSaveName = SaveDataName + "_" + str(rowIndex) + "_" + str(colIndex) + ".png"

                        # 根據機率算要寫哪個檔案
                        Index = np.argmax(WindowProb)

                        # 寫出檔案
                        cv2.imwrite(CurrentSaveName, WindowImg * 255)
                        WriteLine = CurrentSaveName + " " + str(Index) + "\n"
                        DataListSingle.write(WriteLine)

            # 關閉檔案
            DataListSingle.close()

        # 關閉檔案
        DataList.close()
        print("Finish Generating Data!!")

    def _ReadListData(self):
        print("Start Open Data!")

        # 先開起檔案
        DataList = open(self.DataListDir + "/DataList.txt", "r")

        # Data
        DataListStr = DataList.read().split("\n")

        for i in tqdm(range(len(DataListStr) - 1)):          # 跳過最後一個 \n
            DataListSingle = open(DataListStr[i], "r")
            DataListSingleStr = DataListSingle.read().split("\n")
            for j in range(len(DataListSingleStr) - 1):
                CurrentLineData = DataListSingleStr[i].split(" ")

                # 加進資料中
                self.ImgPath.append(CurrentLineData[:-2])
                resultArray = np.zeros([self.OutClass], np.float32)
                if CurrentLineData == 0:
                    self.ZeroIndexArray.append(i)
                else:
                    # 加 Index
                    self.NonZeroIndexArray.append(i)

                # 加 Non Zero 結果
                resultArray[int(CurrentLineData[-1])] = 1
                self.LabelResult.append(resultArray)

        print("Zero Data Size: ", len(self.ZeroIndexArray))
        print("NonZero Data Size: ", len(self.NonZeroIndexArray))
        print("Finish Open Data!")

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