#!/usr/bin/env python
# coding: utf-8

# In[2]:


import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np

#FileLocation = "D:/Dentist/Data/ScanData/2018.11.28/1_origin/"
FileLocation = "D:/Dentist/Data/ScanData/2018.10.18/20181016_Incisor_Label/"
VolumeData = np.zeros([250, 250, 1024], dtype=np.int8)

StartIndex = 60
EndIndex = 200

#Size = (EndIndex - StartIndex + 1) * 250 * 1024
Size = (EndIndex - StartIndex + 1) * 250
print(Size)
x_Value = np.zeros([Size], dtype=np.int8)
y_Value = np.zeros([Size], dtype=np.int8)
z_Value = np.zeros([Size], dtype=np.int8)

for x in range(StartIndex, EndIndex + 1):
    # 讀圖
    TempData = cv2.imread(FileLocation + str(x) + ".png", cv2.IMREAD_GRAYSCALE)
	#TempImage = cv2
    
    for y in range(250):
        x_Value[TempIndex] = x
        y_Value[TempIndex] = y
        for z in range(1024):
            VolumeData[x][y][z] = TempData[y][z]
            
            # 算 Index
            #TempIndex = (x - StartIndex) * 250 * 1024 + y * 1024 + z
			TempIndex = (x - StartIndex) * 250 + y	
			if TempData[x][y][z] == 0:
				z_Value[TempIndex] = z
			


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x_Value, y_Value, z_Value, c = 'b', marker='o')
plt.show()

