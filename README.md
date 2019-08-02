# Outline
---
* [介紹](#Introduction)
* [軟體需求](#軟體需求)
* [如何執行](#Commands)

## 介紹
---
以下為系統流程圖
![系統流程圖](SystemOverview.png)
而底下有幾個程式：
1.  DentistDemo
此專案是 Demo用，將輸入的結果壓縮傳進 DGX Server 裡面
結果預測之後，跑出結果抓下來，顯示出來
2.  DentistProjectV2
主要是所有功能都在裡面
3. DentistProjectV2_TensorflowNetProcess
其中 DentistProjectV2 有使用到 Tensorflow Net
是由此小程式跑出來的結果
4. DentistRawDataReader
這裡主要是 RawData 的測試，在Project 一開始的時候用的，基本上後面的功能都在
DentistProjectV2 裡面都有
5. Paper TPS
主要是做 TPS 的運算 (Python)，以及相關表格繪畫
6. TensorflowNet
網路相關的 Net(Python)
7.  Test_DentistDLL
此專案為測試醫院 OCT(LabelView) 溝通的 DLL 的專案
8. Test_Python
此專案為為測試C++與 Python 溝通的測試專案與 class

## 軟體需求
---
* 共同需要安裝的檔案
Windows Install[連結](https://drive.google.com/drive/folders/16qELBn3ImgEa2IQq6oGBf-WMc3xXDUaw?usp=sharing)
Python3
Matlab 2018a以上，並執行[連結](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)

* Master Branch
1. OtherPackages [連結](https://drive.google.com/file/d/17b6n-TzxYkyxNUnvH5RrwMTeKcDy4P_k/view?usp=sharing)

* SURF_Test Branch
1. OtherPackages_with_xfeature2d [連結](https://drive.google.com/file/d/1pzJ0O5Nb8udP4ZHqwVP2HQob1S7FxNp1/view?usp=sharing)


