import numpy as np
import sys
import math
import cmath

def CircleData(radius):
    centerX = 0.5
    centerY = 0.5
    centerZ = 0.5

    dataX = []
    dataY = []
    dataZ = []

    minX = minY = minZ = sys.maxsize
    maxX = maxY = maxZ = -sys.maxsize

    for x in np.arange(0.3, 0.7, 0.01):
        for y in np.arange(0.3, 0.7, 0.01):
            for z in np.arange(0.3, 0.7, 0.01):
                valX = x.item()
                valY = y.item()
                valZ = z.item()

                if pow(valX - centerX, 2) + pow(valY - centerY, 2) + pow(valZ - centerZ, 2) - pow(radius, 2) < 0.001:
                    dataX.append(valX)
                    dataY.append(valY)
                    dataZ.append(valZ)

    return dataX, dataY, dataZ

def HeartData():
    dataX = []
    dataY = []
    dataZ = []

    minX = minY = minZ = sys.maxsize
    maxX = maxY = maxZ = -sys.maxsize

    for x in np.arange(-3, 3, 0.04):
        for y in np.arange(-3, 3, 0.04):
            for z in np.arange(-3, 3, 0.04):
                valX = x.item()
                valY = y.item()
                valZ = z.item()

                if pow(pow(valX, 2) + 9.0 / 4.0 * pow(valY, 2) + pow(valZ, 2) - 1, 3) - pow(valX, 2) * pow(valZ, 3) - 9.0 / 80.0 * pow(valY, 2) * pow(valZ, 3) < 0.001:
                    dataX.append(valX)
                    dataY.append(valZ)
                    dataZ.append(valY)
                    
    return dataX, dataY, dataZ

def Surface():
    dataX = []
    dataY = []
    dataZ = []

    minX = minY = minZ = sys.maxsize
    maxX = maxY = maxZ = -sys.maxsize

    for x in np.arange(-10, 10, 0.1):
        for y in np.arange(-10, 10, 0.1):
            valX = x.item()
            valY = y.item()
            valZ = (pow(valX, 2) + pow(valY, 2)) / 2.0
            
            dataX.append(valX)
            dataY.append(valZ)
            dataZ.append(valY)

    return dataX, dataY, dataZ

def Tower():
    dataX = []
    dataY = []
    dataZ = []

    minX = minY = minZ = sys.maxsize
    maxX = maxY = maxZ = -sys.maxsize

    for x in np.arange(-20, 20, 0.1):
        for y in np.arange(-20, 20, 0.1):
            valX = x.item()
            valY = y.item()
            valZ = math.sqrt(pow(valX, 2) + pow(valY, 2)) + 3 * math.cos(math.sqrt(pow(valX, 2) + pow(valY, 2)))

            dataX.append(valX)
            dataY.append(valZ)
            dataZ.append(valY)

    return dataX, dataY, dataZ

# 寫檔
def WriteFile(x, y, z):
    assert (len(x) == len(y) and len(y) == len(z))

    file = open("Text.csv", "w")

    # Header
    file.write("X, Y, Z, \n")

    for i in range(len(x)):
        file.write(str(x[i]) + ", " + str(y[i]) + ", " + str(z[i]) + "\n")
    file.close()

x, y, z = Tower()
WriteFile(x, y, z)
