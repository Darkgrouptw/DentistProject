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
__global__ static void findMaxAndMinPeak(float* DataArray, uchar* PointType, int size, int rows, int cols,  float MaxPeakThreshold)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= rows * cols * size)				// 超出範圍
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
	else if (DiffLeft < 0 && DiffRight < 0)
	//else if (DiffLeft < 0 && DiffRight < 0
	//	&& ((-DiffLeft > MinGapPeakThreshold) || (-DiffRight > MinGapPeakThreshold)))
		PointType[id] = 2;
}
__global__ static void ParseMaxMinPeak(uchar* PointType, int size, int rows, int cols, int startIndex)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size * rows)						// 超出範圍
		return;

	// 算 Index
	int sizeIndex = id / rows;
	int rowIndex = id % rows;

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
		int currentID = sizeIndex * rows * cols + rowIndex * cols + i;
		PointType[currentID] = 0;
	}
	for (int i = startIndex; i < cols; i++)
	{
		int currentID = sizeIndex * rows * cols + rowIndex * cols + i;
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
				PointType[sizeIndex * rows * cols + rowIndex * cols + lastMinID] = 0;
				lastMinID = i;
			}
		}
	}

	// 跑到最後結束，要再去判斷最後一個是否是多餘的 Min
	if (lastMinID != -1)
		PointType[sizeIndex * rows * cols + rowIndex * cols + lastMinID] = 0;
}
__global__ static void PickBestChoiceToArray(float* DataArray, uchar* PointType, int* PointType_1D, int size, int rows, int cols, int startIndex,  float Threshold)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size * rows)							// 判斷是否超出大小
		return;

	// 算 Index
	int sizeIndex = id / rows;
	int rowIndex = id % rows;

	bool IsFindMin = false;							// 是否找到底端
	bool IsFindBorder = false;						// 是否找到邊界 (找到底端之後，要開始找邊界)
	float MinData;
	int offsetIndex = sizeIndex * size * cols + rowIndex * cols;
	for (int i = startIndex; i < cols; i++)
	{
		if (PointType[i + offsetIndex] == 2)
		{
			IsFindMin = true;
			MinData = DataArray[i + offsetIndex];
		}
		else if (IsFindMin && DataArray[i + offsetIndex] - MinData > Threshold)
		{
			IsFindBorder = true;
			PointType_1D[sizeIndex * rows + rowIndex] = i;
			break;
		}
	}

	// 接這著要判斷是否找到邊界
	// 如果沒有找到邊界，就回傳 -1
	if (!IsFindBorder)
		PointType_1D[sizeIndex * rows + rowIndex] = -1;

}
__global__ static void ConnectPointsStatus(int * PointType_1D, int* ConnectStatus, int size, int rows, int ConnectRadius)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size * rows)									// 判斷是否超出大小
		return;

	// 算 Index
	int sizeIndex = id / rows;
	int rowIndex = id % rows;

	// 代表這個點沒有有效的點
	if (PointType_1D[sizeIndex * rows + rowIndex] == -1)
		return;

	// 如果是有效的點，就繼續往下追 
	int finalPos = min(rowIndex + ConnectRadius, rows);		// 截止條件
	for (int i = rowIndex + 1; i < finalPos; i++)
	{
		if (PointType_1D[i + sizeIndex * rows] != -1)
		{
			int diffX = PointType_1D[sizeIndex * rows + rowIndex] - PointType_1D[i];
			int diffY = i - rowIndex;
			int Radius = diffX * diffX + diffY * diffY;

			// 0 沒有用到喔
			if (Radius < ConnectRadius * ConnectRadius)
			{
				int index = ConnectRadius * rowIndex + i - rowIndex + sizeIndex * rows;
				ConnectStatus[index] = Radius;
			}
		}
	}
}
__global__ static void TransforToImage(float* DataArray, uchar* OutArray, int size, int rows, int cols)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size * rows * cols)				// 判斷是否超出大小
		return;

	float data = ((DataArray[id] / (float)3.509173f) - (float)(3.39f / 3.509173f)) * 255;
	if (data >= 255)
		OutArray[id] = 255;
	else if (data <= 0)
		OutArray[id] = 0;
	else
		OutArray[id] = (unsigned char)data;
}

//////////////////////////////////////////////////////////////////////////
// CPU
//////////////////////////////////////////////////////////////////////////
void CudaBorder::Init(int size, int rows, int cols)
{
	// 給值
	this->size = size;
	this->rows = rows;
	this->cols = cols;

	// 初始化
	SaveDelete(PointType);
	PointType = new uchar[size * rows * cols];
	memset(PointType, 0, sizeof(uchar) * size * rows * cols);
	SaveDelete(PointType_1D);
	PointType_1D = new int[size * rows];
	memset(PointType_1D, 0, sizeof(int) * size * rows);
}
void CudaBorder::GetBorderFromCuda(float* VolumeData_1D)
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
	float *GPU_VolumeData_1D;
	cudaMalloc(&GPU_VolumeData_1D, sizeof(float) * size * rows * cols);
	cudaMemcpy(GPU_VolumeData_1D, VolumeData_1D, sizeof(float) * size * rows * cols, cudaMemcpyHostToDevice);

	// 點的型別
	uchar* GPU_PointType;
	cudaMalloc(&GPU_PointType, sizeof(uchar) * size * rows * cols);
	cudaMemset(GPU_PointType, 0, sizeof(uchar) * size * rows * cols);
	#pragma endregion
	#pragma region 抓取最大值 每個除以最大值
	float maxValue;
	GetMinMaxValue(VolumeData_1D, maxValue, size * rows * cols);
	NormalizaDataGPU << <NumBlocks, NumThreads >> > (GPU_VolumeData_1D, maxValue, size * rows * cols);
	CheckCudaError();

	// 找最大最小值
	findMaxAndMinPeak << < NumBlocks, NumThreads >> > (GPU_VolumeData_1D, GPU_PointType, size, rows, cols, MaxPeakThreshold);
	CheckCudaError();

	// Parse 一些連續最小值
	ParseMaxMinPeak << < NumBlocks_small, NumThreads_small >> > (GPU_PointType, size, rows, cols, StartIndex);
	CheckCudaError();

	// 抓出一維陣列
	int *GPU_PointType_1D;
	cudaMalloc(&GPU_PointType_1D, sizeof(int) * size * rows);
	PickBestChoiceToArray << <NumBlocks_small, NumThreads_small >> > (GPU_VolumeData_1D, GPU_PointType, GPU_PointType_1D, size, rows, cols, StartIndex, GoThroughThreshold);
	CheckCudaError();

	// 連結點
	int *GPU_Connect_Status;
	cudaMalloc(&GPU_Connect_Status, sizeof(int) * size * rows * ConnectRadius);
	cudaMemset(GPU_Connect_Status, 0, sizeof(int) * size * rows * ConnectRadius);
	ConnectPointsStatus << <NumBlocks_small, NumThreads_small >> > (GPU_PointType_1D, GPU_Connect_Status, size, rows, ConnectRadius);
	CheckCudaError();

	// 把資料傳回 CPU
	int *Connect_Status = new int[size * rows * ConnectRadius];
	cudaMemcpy(PointType,		GPU_PointType,		sizeof(uchar) * size * rows * cols,				cudaMemcpyDeviceToHost);
	cudaMemcpy(PointType_1D,	GPU_PointType_1D,	sizeof(int) * size* rows,						cudaMemcpyDeviceToHost);
	cudaMemcpy(Connect_Status,	GPU_Connect_Status, sizeof(int) * size * rows * ConnectRadius,		cudaMemcpyDeviceToHost);

	GetLargeLine(Connect_Status);
	#pragma endregion
	#pragma region Free Memory
	cudaFree(GPU_VolumeData_1D);
	cudaFree(GPU_PointType);
	cudaFree(GPU_PointType_1D);
	cudaFree(GPU_Connect_Status);

	delete[] Connect_Status;
	#pragma endregion
	#pragma region 結束時間
	time = clock() - time;
	cout << "找邊界: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#pragma endregion
}
vector<Mat> CudaBorder::RawDataToMatArray(float* VolumeData_1D, bool SaveBorder = false)
{
	#pragma region 前置判斷
	// 要先初始化
	assert(PointType_1D != NULL && PointType != NULL && size != 0 && rows != 0 && cols != 0);
	#pragma endregion
	#pragma region 開始時間
	clock_t time;
	time = clock();
	#pragma endregion
	#pragma region 透過	GPU 平行轉值
	// 原 Data Array
	float* GPU_VolumeData_1D;
	cudaMalloc(&GPU_VolumeData_1D, sizeof(float) * size * rows * cols);
	cudaMemcpy(GPU_VolumeData_1D, VolumeData_1D, sizeof(float) * size * rows * cols, cudaMemcpyHostToDevice);

	// Output Uint Array
	// 圖片的資料
	uchar *GPU_UintDataArray, *UintDataArray;
	cudaMalloc(&GPU_UintDataArray, sizeof(uchar) * size * rows * cols);

	// 開始轉圖片
	TransforToImage << <NumBlocks, NumThreads >> > (GPU_VolumeData_1D, GPU_UintDataArray, size, rows, cols);
	CheckCudaError();

	// 轉成 CPU
	UintDataArray = new uchar[size * rows * cols];
	memset(UintDataArray, 0, sizeof(uchar) * size * rows * cols);
	cudaMemcpy(UintDataArray, GPU_UintDataArray, sizeof(uchar) * size * rows * cols, cudaMemcpyDeviceToHost);

	// 轉換到 Mat
	vector<Mat> ImgArray;
	for (int i = 0; i < size; i++)
	{
		// 根據 Offset 拿圖片
		Mat img(rows, cols, CV_8U, UintDataArray + i * rows * cols);
		cvtColor(img.clone(), img, CV_GRAY2BGR);

		// 丟進堆疊
		ImgArray.push_back(img);
	}

	if (SaveBorder)
	{
		// Debug 所有的 peak
		/*for (int i = 0; i < size; i++)
			for (int j = 0; j < rows * cols; j++)
			{
				int offsetIndex = i * rows * cols;
				int rowIndex = j / cols;
				int colIndex = j % cols;

				Vec3b color(0, 0, 0);
				if (PointType[offsetIndex + j] == 1)
					color = Vec3b(0, 255, 255);
				else if (PointType[offsetIndex + j] == 2)
					color = Vec3b(255, 255, 255);
				ImgArray[i].at<Vec3b>(rowIndex, colIndex) = color;
			}*/
		
		// 只抓出最後的邊界
		for (int i = 0; i < size; i++)
			for (int j = 0; j < rows; j++)
			{
				int index = i * rows + j;
				if (PointType_1D[index] != -1)
				{
					Point contourPoint(PointType_1D[index], j);
					circle(ImgArray[i], contourPoint, 2, Scalar(0, 255, 255), CV_FILLED);
				}
			}
	}
	delete[] UintDataArray;
	cudaFree(GPU_UintDataArray);
	cudaFree(GPU_VolumeData_1D);

	// 判斷有無錯誤
	CheckCudaError();
	#pragma endregion
	#pragma region 結束時間
	//time = clock() - time;
	//cout << "轉換圖片時間: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#pragma endregion
	return ImgArray;
}

void CudaBorder::GetMinMaxValue(float* begin, float& max, int size)
{
	clock_t time = clock();
	float* iter = max_element(begin, begin + size);
	unsigned int position = iter - begin;
	float max_val = *iter;
	//cout << "最大值是: " << max_val << " 在位置: " << position << endl;
	max = max_val;
}
void CudaBorder::GetLargeLine(int *Connect_Status)
{
	for (int i = 0; i < size; i++)
	{
		// 每個 10 段下去 Sample
		int RowGap = rows / 10;
		vector<vector<int>> StatusVector;
		for (int j = 0; j < rows; j += RowGap)
		{
			int begin = j;
			int end = j;

			if (PointType_1D[j + i * rows] == -1)
				continue;

			// 往上找 & 先加上自己
			vector<int> Connect;
			Connect.push_back(j);

			int FindIndex = j;
			bool IsFind = true;
			while (IsFind && FindIndex > 0)
			{
				int minIndex = -1;
				int tempValue = ConnectRadius * ConnectRadius;
				for (int k = 1; k < ConnectRadius; k++)
				{
					int index = ConnectRadius * (FindIndex - k) + k + i * rows * ConnectRadius;
					if (FindIndex - k >= 0 && Connect_Status[index] != 0 && tempValue > Connect_Status[index])
					{
						tempValue = Connect_Status[index];
						minIndex = k;
					}
				}

				if (minIndex != -1)
				{
					FindIndex = FindIndex - minIndex;
					Connect.push_back(FindIndex);
					IsFind = true;
				}
				else
					IsFind = false;
			}

			// 往下找
			FindIndex = j;
			while (IsFind && FindIndex < rows - 1)
			{
				int minIndex = -1;
				int tempValue = ConnectRadius * ConnectRadius;
				for (int k = 1; k < ConnectRadius; k++)
				{
					int index = ConnectRadius * FindIndex + k + i * rows * ConnectRadius;
					if (FindIndex + k < rows && Connect_Status[index] != 0 && tempValue > Connect_Status[index])
					{
						tempValue = Connect_Status[index];
						minIndex = k;
					}
				}

				if (minIndex != -1)
				{
					FindIndex = FindIndex + minIndex;
					Connect.push_back(FindIndex);
					IsFind = true;
				}
				else
					IsFind = false;
			}

			if (Connect.size() > 1)
			{
				// 由小排到大
				sort(Connect.begin(), Connect.end());
				StatusVector.push_back(Connect);
			}
		}

		// 前面的幾個張數，可能會找不到點，所以跳過處理
		if (StatusVector.size() == 0)
			continue;

		// 排序之後取最大
		sort(StatusVector.begin(), StatusVector.end(), SortByVectorSize);

		// 把其他雜點刪掉
		vector<int> LineVector = StatusVector[0];
		int index = 0;
		for (int j = 0; j < rows; j++)
		{
			if (LineVector[index] != i)
				PointType_1D[j + i * rows] = -1;
			else if (LineVector[index] == i)
			{
				index++;
				if (index >= LineVector.size())
				{
					for (int k = j + 1; k < rows; k++)
						PointType_1D[k + i * rows] = -1;
					break;
				}
			}
		}
	}
}
bool CudaBorder::SortByVectorSize(vector<int> left, vector<int> right)
{
	return right.size() < left.size();
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
		delete[] pointer;
}
