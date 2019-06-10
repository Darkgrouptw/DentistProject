/*
加速用的工具
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

class UtilityTools
{
public:
	UtilityTools();
	~UtilityTools();

	//////////////////////////////////////////////////////////////////////////
	// 轉換 Function
	//////////////////////////////////////////////////////////////////////////
	void SetImageData(vector<Mat>, int, int, int, int);
	vector<Mat> TransfromMatArray();

private:
	//////////////////////////////////////////////////////////////////////////
	// 圖片資料
	//////////////////////////////////////////////////////////////////////////
	uchar3 *CPUImageData = NULL;													// 圖片資訊
	int size;																	// 慢軸 (SizeY)
	int rows;																	// 快軸 (SizeX)
	int cols;																	// 深度 (Size / 2)

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	inline void CheckCudaError();												// 判斷 Cuda 有無 Error
	void SaveDelete(void* pointer);

	//////////////////////////////////////////////////////////////////////////
	// 其他參數設定 設定
	//////////////////////////////////////////////////////////////////////////
	const int NumThreads = 1024;												// 最多限制在 1024
	const int SmoothSizeRange = 11;												// 向外 Smooth  左 (SmoothLength - 1) / 2 + 中間 1 + 右 (SmoothLength - 1) / 2

};