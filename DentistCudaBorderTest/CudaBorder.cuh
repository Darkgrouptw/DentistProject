#include <iostream>
#include <ctime>

#include <QImage>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include <algorithm>

using namespace std;

class CudaBorder
{
public:
	CudaBorder();
	~CudaBorder();

	void Init(int, int);													// 初始化
	void GetBorderFromCuda(float **);										// 抓出邊界，最後面要回傳出邊界
	QImage SaveDataToImage(float**);										// 這邊要存出圖片，然後

	//////////////////////////////////////////////////////////////////////////
	// 外部存取資料
	//////////////////////////////////////////////////////////////////////////
	uchar* PointType = NULL;												// 存成 2D 每個點是什麼 Type (0 無、1 Max Peak、 2 Min Peak)
	int* PointType_1D = NULL;												// 存哪一個為 Border

private:
	//////////////////////////////////////////////////////////////////////////
	// 相關變數
	//////////////////////////////////////////////////////////////////////////
	int rows = 0;															// height
	int cols = 0;															// width

	//////////////////////////////////////////////////////////////////////////
	// 處理相關
	//////////////////////////////////////////////////////////////////////////
	void GetMinMaxValue(float *, float&, int);								// 抓取最大最小值 (為了做 Normalize)
	void NormalizeData(float *, float, int);								// 根據最大值，去壓縮資料大小
	void GetLargeLine(int *);												// 取出最大條的線，並刪除其他雜點
	static bool SortByVectorSize(vector<int>, vector<int>);						// 排序

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	inline void CheckCudaError();
	inline void SaveDelete(void*);

	//////////////////////////////////////////////////////////////////////////
	// 參數設定 設定
	//////////////////////////////////////////////////////////////////////////
	const int NumBlocks = 512;
	const int NumThreads = 512;
	const int NumBlocks_small = 16;
	const int NumThreads_small = 16;
	const float MaxPeakThreshold = 0.7f;									// 要高於這個值
	//const float MinGapPeakThreshold = 0.01f;								// 間隔要超過這個差距
	const float GoThroughThreshold = 0.08f;									// 要多少在走過去
	const int StartIndex = 10;												// 從這裡開始找有效的資料
	const int ConnectRadius = 25;											// 連結半徑
};