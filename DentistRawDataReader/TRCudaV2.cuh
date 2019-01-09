#include <iostream>
#include <vector>
#include <ctime>
#include <cassert>
#include <numeric>
#include <algorithm>

#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h> 

using namespace std;

typedef unsigned short ushort;
typedef unsigned char uchar;

class TRCudaV2
{
public:
	TRCudaV2();
	~TRCudaV2();

	// 轉換 Function
	void RawDataToPointCloud(char*, int, int, int ,int, long, double, int);

	int *OCTData = NULL;

private:
	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	inline void CheckCudaError();
	inline void SaveDelete(void*);

	//////////////////////////////////////////////////////////////////////////
	// 硬體設定的 Function
	//////////////////////////////////////////////////////////////////////////

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