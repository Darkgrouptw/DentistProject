﻿/*
這邊主要是作將 Raw Data 轉成圖的部分
且邊界判斷也在這做
*/
#include "GlobalDefine.h"

#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <cassert>
#include <numeric>
#include <algorithm>

#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define generic __identifier(generic)
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#undef generic

using namespace std;
using namespace cv;

typedef unsigned short ushort;
typedef unsigned char uchar;

// 連接時，會用這個資料型態
struct ConnectInfo
{
	int rowIndex;																// 第幾個 row
	int chooseIndex;															// 選到第幾個
};

class TRCudaV2
{
public:
	TRCudaV2();
	~TRCudaV2();

	//////////////////////////////////////////////////////////////////////////
	// 轉換 Function
	//////////////////////////////////////////////////////////////////////////
	void SingleRawDataToPointCloud(char*, int, int, int, long, double, int);	// 這邊應該只有 250 * 2048
	void MultiRawDataToPointCloud(char*, int, int, int ,int, long, double, int);// 250 * 250 * 2048

	//////////////////////////////////////////////////////////////////////////
	// 拿出圖片
	//////////////////////////////////////////////////////////////////////////
	vector<Mat> TransfromMatArray(bool);										// 這邊要存出圖片 (bool 是是否要儲存出邊界資訊)
	Mat TransformToOtherSideView();												// TopView
	void CopySingleBorder(int*);												// Copy 單張 Border 的
	void CopyBorder(int*);														// Copy 整段的 Border
	bool ShakeDetect_Single(int*, bool);										// 晃動偵測 (傳入前一張的 Single)	=> True 代表有晃動
	bool ShakeDetect_Multi(bool, bool);											// 晃動偵測，是否要用較精確的	Threshold => True 代表有晃動

	int* TestBestN;
private:
	//////////////////////////////////////////////////////////////////////////
	// 圖片資料
	//////////////////////////////////////////////////////////////////////////
	uchar *VolumeData = NULL;													// 圖片資訊
	uchar *VolumeData_OtherSide = NULL;											// TopView 
	uchar* PointType = NULL;													// 存成 2D 每個點是什麼 Type (0 無、1 Max Peak、 2 Min Peak)
	int* PointType_1D = NULL;													// 存哪一個為 Border
	int size;																	// 慢軸 (SizeY)
	int rows;																	// 快軸 (SizeX)
	int cols;																	// 深度 (Size / 2)

	//////////////////////////////////////////////////////////////////////////
	// 計時相關
	//////////////////////////////////////////////////////////////////////////
	clock_t time;																// 一個步驟的時間
	clock_t totalTime;															// 總共步驟的時間

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	void GetSurface(int *PointType_BestN, int *Connect_Status);					// 抓取表面的線 (可能中間會斷掉)
	static bool SortByRows(ConnectInfo, ConnectInfo);							// 依照 Row 小到大排序
	static bool SortByVectorSize(vector<ConnectInfo>, vector<ConnectInfo>);		// 排序 Vector Size (由大到小)
	inline void CheckCudaError();												// 判斷 Cuda 有無 Error
	inline void SaveDelete(void*);												// Save Delete CPU 變數

	//////////////////////////////////////////////////////////////////////////
	// 找邊界參數設定
	//////////////////////////////////////////////////////////////////////////
	const int SmoothSizeRange = 15;												// 向外 Smooth  左 (SmoothLength - 1) / 2 + 中間 1 + 右 (SmoothLength - 1) / 2
	const float MaxPeakThreshold = 0.08f;										// 要高於這個值
	const float GoThroughThreshold = 0.05f;										// 要多少在走過去
	const float SatPeakThreshold = 200;											// 這個是用來刪除過飽和的 Threshold
	const int ChooseBestN = 3;
	const int StartIndex = 10;													// 從這裡開始找有效的資料
	const int DenoiseWindowsRadius = 10;										// 幾個 Pixel 以內的找最多的點
	const int ConnectRadius = 50;												// 連結半徑

	//////////////////////////////////////////////////////////////////////////
	// 晃動 Threshold
	//////////////////////////////////////////////////////////////////////////
	const float OCT_Move_Threshold = 20;										// 晃動大於某一個值，代表不能用
	const float OCT_Move_Precise_Threshold = 10;								// 晃動大於某一個值，代表不能用 (較精確)
	const int	OCT_Valid_VoteNum = 20;											// 有效票數起碼要大於這個值

	//////////////////////////////////////////////////////////////////////////
	// 其他參數設定 設定
	//////////////////////////////////////////////////////////////////////////
	const int NumThreads = 1024;												// 最多限制在 1024
	const int NumPolynomial = 5;												// 使用 5 次項
	const int MinValuePixel_TL = 2;												// 再算最小值的時候，是根據一塊確定沒有資料的部分，去算 MinValue，而這個是 Top Left 的 Pixel
	const int MinValuePixel_BR = 10;											// Buttom Right 的 Pixel
	const float MinValueScalar = 1.75f;											// 由於有大部分都是雜訊，用這個可以濾掉建議 1.8 ~ 2 之間 (測試用)
	const float OtherSideMean = 50;												// 由於硬體的大小改變，會導至判斷上會有問題
};