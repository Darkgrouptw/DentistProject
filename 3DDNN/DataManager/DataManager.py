import numpy as np
import cv2

class DataManager:
    def __init__(
            self,
            FileNameList,
            LabeledList,
            OutClass,
            Ratio = 0.9
    ):
        # 前置判斷
        print("DataSize: ", len(FileNameList))
        print("LabeledSize: ", len(LabeledList))
        assert(len(FileNameList) == len(LabeledList))
        self.DataSize = len(FileNameList)
        self.OutClass = OutClass

        # 讀檔案
        self._ReadData(FileNameList, LabeledList)

        # 參數設定
        self.TrainValidRatio = Ratio


    # 把資料載進來
    def _ReadData(self, FileNameList, LabeledList):
        # 讀 Input
        for i in range(self.DataSize):
            file = open(FileNameList[i])
            contents = file.readlines()

            # 根據每行下去讀
            for j in range(len(contents)):
                data = contents[j].split(", ")

                # 拿最大邊界，並產生資料的 Array
                if j == 0 and i == 0:
                    for k in range(3):
                        MaxValue = data[k].split(" ")[1].replace(")", "")
                        if k == 0:
                            self.Cols = int(MaxValue)
                        elif k == 1:
                            self.Rows = int(MaxValue)
                        else:
                            self.Depth = int(MaxValue)
                    self.Data = np.zeros([self.DataSize, self.Depth, self.Rows, self.Cols], dtype=np.float32)
                elif j != 0:
                    x = int(data[0])
                    y = int(data[1])
                    z = int(data[2])
                    self.Data[i, z, y, x] = 1
        print("Finish Load csv!")

        # 讀取圖片
        for i in range(self.DataSize):
            LabelImg = cv2.imread(LabeledList[i], cv2.IMREAD_GRAYSCALE)

            # 初始化陣列大小
            if i == 0:
                self.LabelData = np.zeros([self.DataSize, self.Rows, self.Cols, self.OutClass], dtype=np.float32)

            # 先拿出 255 & 0 的 Index
            LabelImg_OneHot = np.zeros([self.Rows, self.Cols, self.OutClass])
            LabelImg_Temp0 = np.asarray((LabelImg == 0).nonzero())
            LabelImg_Temp1 = np.asarray((LabelImg == 255).nonzero())

            # 轉 One Hot 的圖
            LabelImg_OneHot[LabelImg_Temp0[0], LabelImg_Temp0[1], 0] = 1
            LabelImg_OneHot[LabelImg_Temp1[0], LabelImg_Temp1[1], 1] = 1
            self.LabelData[i] = LabelImg_OneHot
        print("Fininsh Load Image")


    # 拿一部分的 Train Data
    def BatchTrainData(self, size):
        choice = np.random.randint(int(self.DataSize * self.TrainValidRatio), size=size)
        return self.Data[choice].reshape(size, self.Depth, self.Rows, self.Cols, 1), self.LabelData[choice].reshape(size, self.Rows, self.Cols, 2)

    # 拿 Valid Data
    def BatchValidData(self, size):
        choice = np.random.randint(int(self.DataSize * (1 - self.TrainValidRatio)), size=size) + int(self.DataSize * self.TrainValidRatio)
        return self.Data[choice].reshape(size, self.Depth, self.Rows, self.Cols, 1), self.LabelData[choice].reshape(size, self.Rows, self.Cols, 1)

    # 測試一張圖
    def TestFirstBoxOfData(self, size):
        return self.Data[:size].reshape(size, self.Depth, self.Rows, self.Cols, 1)
