import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def CircleData(BoxSize, CenterPos, radius):
    dataX = []
    dataY = []
    dataZ = []

    # 圖片
    Answer = np.zeros([BoxSize[0], BoxSize[1], 3], dtype=np.uint8)

    # 設定中心點
    centerX = CenterPos[0]
    centerY = CenterPos[1]
    centerZ = CenterPos[2]

    for x in np.arange(0, BoxSize[0]):
        for y in np.arange(0, BoxSize[1]):
            CanBeProjection = False
            for z in np.arange(0, BoxSize[2]):
                valX = x.item()
                valY = y.item()
                valZ = z.item()

                if pow(valX - centerX, 2) + pow(valY - centerY, 2) + pow(valZ - centerZ, 2) - pow(radius, 2) < 0.001:
                    dataX.append(valX)
                    dataY.append(valY)
                    dataZ.append(valZ)
                    CanBeProjection = True

            # 能投影到就是 1
            if CanBeProjection:
                Answer[x, y, 0] = 255
                Answer[x, y, 1] = 255
                Answer[x, y, 2] = 255
    return dataX, dataY, dataZ, Answer

# 寫檔
def WriteFile(BoxSize, Ans, Title, LabelTitle, x, y, z, CenterPos, Radius):
    assert (len(x) == len(y) and len(y) == len(z))

    # File Header
    file = open(Title, "w")
    file.write("X(0 " + str(BoxSize[0]) + "), Y(0 " + str(BoxSize[1]) + "), Z(0 " + str(BoxSize[2]) + "), " + str(CenterPos[0]) + " " + str(CenterPos[1]) + " " + str(CenterPos[2]) + ", " + str(Radius) + " \n")

    for i in range(len(x)):
        file.write(str(x[i]) + ", " + str(y[i]) + ", " + str(z[i]) + "\n")
    file.close()

    # Answer
    # file = open("Ans.csv", "w")
    # file.write("x")
    # file.close()
    plt.imsave(LabelTitle, Ans)

# Main
# BoxSize = np.array([250, 250, 1024], np.int32)
Radius = 20
BoxSize = np.array([Radius * 2, Radius * 2, Radius * 2], np.int32)
CenterPos = np.array([Radius, Radius, Radius], np.int32)

DataSetSize = 1000

for i in tqdm(range(0, DataSetSize)):
    MovePos = np.random.randint(-Radius, Radius, 3)
    CenterPosTemp = CenterPos + MovePos

    RadiusRand = np.random.randint(0, Radius)
    x, y, z, ans = CircleData(BoxSize, CenterPosTemp, 1 + RadiusRand)

    CircleFileName = "./Data/Circle_" + str(i) + ".csv"
    LabelFileName = "./Data/AnsCircle_" + str(i) + ".png"
    WriteFile(BoxSize, ans, CircleFileName, LabelFileName, x, y, z, CenterPosTemp, 1 + RadiusRand)