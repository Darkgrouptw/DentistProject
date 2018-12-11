#include "CudaBorder.cuh"

CudaBorder::CudaBorder()
{
}
CudaBorder::~CudaBorder()
{
	SaveDelete(PointType);
	SaveDelete(PointType_1D);
}

//////////////////////////////////////////////////////////////////////////
// GPU
//////////////////////////////////////////////////////////////////////////
__global__ static void NormalizaDataGPU(float* DataArray, float maxValue, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size)
		return;

	DataArray[id] /= maxValue;
}
__global__ static void findMaxAndMinPeak(float* DataArray, uchar* PointType, int rows, int cols, float MinGapPeakThreshold,  float MaxPeakThreshold)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= rows * cols)				// 超出範圍
		return;

	// width 判斷 1 ~ (width - 1)
	int colID = id % cols;
	if (1 >= colID || colID == (cols - 1))
		return;

	// 接著要去比周圍
	// 峰值判斷 (要比兩邊高，且峰值要高於某一個值，且左 或右差值，只有一端能高於這個值)
	float DiffLeft = DataArray[id] - DataArray[id - 1];
	float DiffRight = DataArray[id] - DataArray[id + 1];
	if (DiffLeft > 0 && DiffRight > 0
		&& DataArray[id] > MaxPeakThreshold)
		PointType[id] = 1;
	else if (DiffLeft < 0 && DiffRight < 0
		&& ((-DiffLeft > MinGapPeakThreshold) || (-DiffRight > MinGapPeakThreshold)))
		PointType[id] = 2;
}
__global__ static void ParseMaxMinPeak(uchar* PointType, int rows, int cols, int startIndex)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= rows)						// 超出範圍
		return;

	// 然後這邊要去 Skip 所有的 Min
	int lastMinID = -1;
	bool FindMax = false;

	// 為了要抓出 最大(有效)的 區間
	int Useful_Start = -1;
	int Useful_End = -1;
	int Useful_PeakCount = -1, tempPeakCount = 0;

	// 刪除多餘 min Peak
	for (int i = 0; i < startIndex; i++)
	{
		int currentID = id * cols + i;
		PointType[currentID] = 0;
	}
	for (int i = startIndex; i < cols; i++)
	{
		int currentID = id * cols + i;
		if (lastMinID == -1)			// 判斷是不適剛開始 or 找到 Max
		{
			// 要先去抓出第一個 Min
			if (PointType[currentID] == 2)
				lastMinID = i;
			else if (PointType[currentID] == 1)
				PointType[currentID] = 0;				// 這邊代表沒有遇到峰值，應該是雜訊了
		}
		else
		{
			// 已經抓到 min 了之後，要去濾掉其他的 min
			if (PointType[currentID] == 1)
			{
				// 抓到 Max
				FindMax = true;
				tempPeakCount++;
			}
			else if (FindMax && PointType[currentID] == 2)
			{
				// 抓到 Max 之後，又找到一個 Min
				if (Useful_PeakCount < tempPeakCount)
				{
					Useful_PeakCount = tempPeakCount;
					Useful_Start = lastMinID;
					Useful_End = i;
				}
				FindMax = false;
				tempPeakCount = 0;
				lastMinID = -1;
			}
			else if (!FindMax && PointType[currentID] == 2)
			{
				// 沒抓到 Max 只抓到 Min
				PointType[id * cols + lastMinID] = 0;
				lastMinID = i;
			}
		}
	}

	// 跑到最後結束，要再去判斷最後一個是否是多餘的 Min
	if (lastMinID != -1)
		PointType[id * cols + lastMinID] = 0;
}
__global__ static void TransforToImage(float* DataArray, uchar* OutArray, int rows, int cols)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= rows * cols)				// 判斷是否超出大小
		return;

	float data = ((DataArray[id] / (float)3.509173f) - (float)(3.39f / 3.509173f)) * 255;
	if (data >= 255)
		OutArray[id] = 255;
	else if (data <= 0)
		OutArray[id] = 0;
	else
		OutArray[id] = (unsigned char)data;
}
__global__ static void PickBestChoiceToArray(float* DataArray, uchar* PointType, int* PointType_1D, int rows, int cols, float Threshold)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= rows)									// 判斷是否超出大小
		return;

	bool IsFindMin = false;							// 是否找到底端
	bool IsFindBorder = false;						// 是否找到邊界 (找到底端之後，要開始找邊界)
	float MinData;
	int offsetIndex = id * cols;
	//printf("%d\n", id);
	for (int i = 0; i < cols; i++)
	{
		if (!IsFindMin && PointType[i + offsetIndex] == 2)
		{
			IsFindMin = true;
			MinData = DataArray[i + offsetIndex];
		}
		else if (IsFindMin && DataArray[i + offsetIndex] - MinData > Threshold)
		{
			IsFindBorder = true;
			PointType_1D[id] = i;
			break;
		}
	}

	// 接這著要判斷是否找到邊界
	// 如果沒有找到邊界，就回傳 -1
	if (!IsFindBorder)
		PointType_1D[id] = -1;

}
__global__ static void StablizePointType(int * PointType_1D, int* OutputType_1D, int rows, int chooseSize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= rows)									// 判斷是否超出大小
		return;

	// 由於雜點會有很多 Noise 要去雜訊用的
	int range = chooseSize / 2;
	int count = 0;
	int TotalHeight = 0;
	for (int i = -range; i <= range; i++)
	{
		if (0 > i + id || id + i >= rows)			// 超出邊界
			continue;
		else if (PointType_1D[id + i] == -1)		// 沒有資訊
			continue;
		else if (abs(PointType_1D[id + i] - PointType_1D[id]) > 30)		// 差距太大
			continue;
		else
		{
			TotalHeight += PointType_1D[id + i];
			count++;
		}
	}

	if (count >= range - 1)
		OutputType_1D[id] = TotalHeight / count;
	else
		OutputType_1D[id] = -1;
}

//////////////////////////////////////////////////////////////////////////
// CPU
//////////////////////////////////////////////////////////////////////////
void CudaBorder::Init(int rows, int cols)
{
	// 給值
	this->rows = rows;
	this->cols = cols;

	// 初始化
	SaveDelete(PointType);
	PointType = new uchar[rows * cols];
	SaveDelete(PointType_1D);
	PointType_1D = new int[rows];
}
void CudaBorder::GetBorderFromCuda(float** DataArray)
{
	#pragma region 前置判斷
	// 要先初始化
	assert(PointType_1D != NULL && PointType != NULL && rows != 0 && cols != 0);
	#pragma endregion
	#pragma region 開始時間
	clock_t time;
	time = clock();
	#pragma endregion
	#pragma region GPU Init
	float *GPU_DataArray;
	cudaMalloc(&GPU_DataArray, sizeof(float) * rows * cols);
	cudaMemcpy(GPU_DataArray, &DataArray[0][0], sizeof(float) * rows * cols, cudaMemcpyHostToDevice);

	// 點的型別
	uchar* GPU_PointType;
	cudaMalloc(&GPU_PointType, sizeof(uchar) * rows * cols);
	cudaMemset(GPU_PointType, 0, sizeof(uchar) * rows * cols);
	#pragma endregion
	#pragma region 抓取最大值 每個除以最大值
	float maxValue = GetMaxValue(GPU_DataArray,  rows * cols);
	NormalizeData(GPU_DataArray, maxValue, rows * cols);

	// 找最大最小值
	findMaxAndMinPeak << < NumBlocks, NumThreads >> > (GPU_DataArray, GPU_PointType, rows, cols, MinGapPeakThreshold, MaxPeakThreshold);
	CheckCudaError();

	// Parse 一些連續最小值
	ParseMaxMinPeak << < NumBlocks_small, NumThreads_small >> > (GPU_PointType, rows, cols, StartIndex);
	CheckCudaError();

	// 抓出一維陣列
	int *GPU_PointType_1D, *GPU_tempPointType_1D;
	cudaMalloc(&GPU_PointType_1D, sizeof(int) * rows);
	cudaMalloc(&GPU_tempPointType_1D, sizeof(int) * rows);
	PickBestChoiceToArray << <NumBlocks_small, NumThreads_small >> > (GPU_DataArray, GPU_PointType, GPU_tempPointType_1D, rows, cols, 0.03f);
	CheckCudaError();

	StablizePointType << <NumBlocks_small, NumThreads_small >> > (GPU_tempPointType_1D, GPU_PointType_1D, rows, StablizeSize);
	CheckCudaError();

	// 把資料傳回 CPU
	cudaMemcpy(PointType, GPU_PointType, sizeof(uchar) * rows * cols, cudaMemcpyDeviceToHost);
	cudaMemcpy(PointType_1D, GPU_PointType_1D, sizeof(int) * rows, cudaMemcpyDeviceToHost);
	//cudaMemcpy(PointType_1D, GPU_tempPointType_1D, sizeof(int) * rows, cudaMemcpyDeviceToHost);
	#pragma endregion
	#pragma region Free Memory
	cudaFree(GPU_DataArray);
	cudaFree(GPU_PointType);
	cudaFree(GPU_PointType_1D);
	cudaFree(GPU_tempPointType_1D);
	#pragma endregion
	#pragma region 結束時間
	time = clock() - time;
	cout << "找最大值: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#pragma endregion
}
QImage CudaBorder::SaveDataToImage(float** DataArray)
{
	#pragma region 前置判斷
	// 要先初始化
	assert(PointType_1D != NULL && PointType != NULL  && rows != 0 && cols != 0);
	#pragma endregion
	#pragma region 開始時間
	clock_t time;
	time = clock();
	#pragma endregion
	#pragma region 透過	GPU 平行轉值
	// 原 Data Array
	float* GPU_DataArray;
	cudaMalloc(&GPU_DataArray, sizeof(float) * rows * cols);
	cudaMemcpy(GPU_DataArray, &DataArray[0][0], sizeof(float) * rows * cols, cudaMemcpyHostToDevice);

	// Output Uint Array
	uchar *GPU_UintDataArray, *UintDataArray;
	cudaMalloc(&GPU_UintDataArray, sizeof(uchar) * rows * cols);

	// 開始轉圖片
	TransforToImage << <NumBlocks, NumThreads >> > (GPU_DataArray, GPU_UintDataArray, rows, cols);
	CheckCudaError();

	// 轉成 CPU
	UintDataArray = new unsigned char[rows * cols];
	memset(UintDataArray, 0, sizeof(unsigned char) * rows * cols);
	cudaMemcpy(UintDataArray, GPU_UintDataArray, sizeof(unsigned char) * rows * cols, cudaMemcpyDeviceToHost);

	// 轉換到 QImage
	QImage img(UintDataArray, cols, rows, QImage::Format_Grayscale8);
	img = img.convertToFormat(QImage::Format_RGB888);
	// Debug 所有的 peak
	//for (int i = 0; i < rows * cols; i++)
	//{
	//	int rowIndex = i / cols;
	//	int colIndex = i % cols;

	//	QColor color(0, 0, 0);
	//	/*if (PointType[i] == 1)
	//		color = QColor(255, 255, 0);
	//	else */
	//	if (PointType[i] == 2)
	//	{
	//		color = QColor(255, 255, 255);
	//		img.setPixelColor(colIndex, rowIndex, color);
	//	}

	//}
	// 這邊是最後抓出來的邊界
	QColor color(255, 255, 0);
	for (int i = 0; i < rows; i++)
		if (PointType_1D[i] != -1)
			img.setPixelColor(PointType_1D[i], i, color);
	delete UintDataArray;
	cudaFree(GPU_UintDataArray);

	// 判斷有無錯誤
	CheckCudaError();
	#pragma endregion
	#pragma region 結束時間
	time = clock() - time;
	cout << "轉換圖片時間: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#pragma endregion
	return img;
}

float CudaBorder::GetMaxValue(float* GPU_begin, int size)
{
	thrust::device_vector<float> d_vec(GPU_begin, GPU_begin + size);
	thrust::device_vector<float>::iterator iter = thrust::max_element(thrust::device, d_vec.begin(), d_vec.end());

	unsigned int position = iter - d_vec.begin();
	float max_val = *iter;

	CheckCudaError();

	cout << "最大值是: " << max_val << " 在位置: " << position << endl;
	return max_val;
}
void CudaBorder::NormalizeData(float *GPU_DataArray, float maxValue, int size)
{
	NormalizaDataGPU << <NumBlocks, NumThreads >> > (GPU_DataArray, maxValue, size);
	CheckCudaError();
}

//////////////////////////////////////////////////////////////////////////
// Helper Function
//////////////////////////////////////////////////////////////////////////
void CudaBorder::CheckCudaError()
{
	cudaError GPU_Error = cudaGetLastError();
	if (GPU_Error != cudaSuccess)
	{
		cout << cudaGetErrorString(GPU_Error) << endl;
		assert(false);
		exit(-1);
	}
}
void CudaBorder::SaveDelete(void* pointer)
{
	if (pointer != NULL)
		delete pointer;
}
