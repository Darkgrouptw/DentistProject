#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <ctime>
#include <cassert>
#include <numeric>
#include <algorithm>

#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h> 

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

using namespace std;
using namespace cv;

typedef unsigned short ushort;
typedef unsigned char uchar;

class TRCudaV2
{
public:
	TRCudaV2();
	~TRCudaV2();

	// 轉換 Function
	void SingleRawDataToPointCloud(char*, int, int, int, long, double, int);	// 這邊應該只有 250 * 2048
	void RawDataToPointCloud(char*, int, int, int ,int, long, double, int);		// 250 * 250 * 2048

	// 拿出圖片
	vector<Mat> TransfromMatArray(bool);										// 這邊要存出圖片 (bool 是是否要儲存出邊界資訊)

private:
	//////////////////////////////////////////////////////////////////////////
	// 圖片資料
	//////////////////////////////////////////////////////////////////////////
	uchar *VolumeData = NULL;													// 圖片資訊
	//uchar* PointType = NULL;													// 存成 2D 每個點是什麼 Type (0 無、1 Max Peak、 2 Min Peak)
	//int* PointType_1D = NULL;													// 存哪一個為 Border
	int size;																	// 慢軸 (SizeY)
	int rows;																	// 快軸 (SizeX)
	int cols;																	// 深度 (Size / 2)

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	inline void CheckCudaError();
	inline void SaveDelete(void*);

	//////////////////////////////////////////////////////////////////////////
	// 參數設定 設定
	//////////////////////////////////////////////////////////////////////////
	//const int NumBlock_Full
	const int NumBlocks = 250 * 250;
	const int NumThreads = 1024;											// 最多限制在 1024
	//const int NumBlocks_small = 250 * ChooseBestN;
	//const int NumThreads_small = 250;
	const int NumPolynomial = 5;											// 使用 5 次項
};