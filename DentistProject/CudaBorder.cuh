#include <iostream>
#include <vector>
#include <ctime>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h> 

using namespace std;
using namespace cv;

// 連接時，會用這個資料型態
struct ConnectInfo 
{
	int rowIndex;															// 第幾個 row
	int chooseIndex;														// 選到第幾個
};

class CudaBorder
{
public:
	CudaBorder();
	~CudaBorder();

	void Init(int, int, int);												// 初始化
	void GetBorderFromCuda(float *);										// 抓出邊界，最後面要回傳出邊界
	vector<Mat> RawDataToMatArray(float*, bool);							// 這邊要存出圖片 (bool 是是否要儲存出邊界資訊)

	//////////////////////////////////////////////////////////////////////////
	// 外部存取資料
	//////////////////////////////////////////////////////////////////////////
	uchar* PointType = NULL;												// 存成 2D 每個點是什麼 Type (0 無、1 Max Peak、 2 Min Peak)
	int* PointType_1D = NULL;												// 存哪一個為 Border

private:
	//////////////////////////////////////////////////////////////////////////
	// 相關變數
	//////////////////////////////////////////////////////////////////////////
	int size = 0;															// 有幾張
	int rows = 0;															// height
	int cols = 0;															// width

	//////////////////////////////////////////////////////////////////////////
	// 處理相關
	//////////////////////////////////////////////////////////////////////////
	void GetMinMaxValue(float *, float&, int);								// 抓取最大最小值 (為了做 Normalize)
	void GetLargeLine(int *, int *);										// 取出最大條的線，並刪除其他雜點
	static bool SortByRows(ConnectInfo, ConnectInfo);						// 依照 Row 小到大排序
	static bool SortByVectorSize(vector<ConnectInfo>, vector<ConnectInfo>);	// 排序 Vector Size (由大到小)

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	inline void CheckCudaError();
	inline void SaveDelete(void*);

	//////////////////////////////////////////////////////////////////////////
	// 參數設定 設定
	//////////////////////////////////////////////////////////////////////////
	const float MaxPeakThreshold = 0.67f;									// 要高於這個值
	const float GoThroughThreshold = 0.01f;									// 要多少在走過去
	const int ChooseBestN = 3;
	//const int StartIndex = 10;											// 從這裡開始找有效的資料
	const int StartIndex = 150;												// 從這裡開始找有效的資料
	const int MinGroupSize = 20;											// 最少要有幾個點
	const int ConnectRadius = 25;											// 連結半徑
	const int NumBlocks = 250 * 250;
	const int NumThreads = 1024;											// 聽說最多限制在 1024
	const int NumBlocks_small = 250 * ChooseBestN;
	const int NumThreads_small = 250;
};