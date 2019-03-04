import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 設定 Label
def Init3DPlot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Size (X)')
    ax.set_ylabel("Pic (Y)")
    ax.set_zlabel('Depth (Z)')
    return ax

def CircleData():
    dataX = []
    dataY = []
    dataZ = []

    return dataX, dataY, dataZ


ax = Init3DPlot()
plt.show()