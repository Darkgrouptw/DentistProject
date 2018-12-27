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
__global__ static void PickBestChoiceToArray(float* DataArray, uchar* PointType, int* PointType_BestN, int size, int rows, int cols, int ChooseBestN, int startIndex,  float Threshold)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size * rows)							// 判斷是否超出大小
		return;

	// 算 Index
	int sizeIndex = id / rows;
	int rowIndex = id % rows;

	bool IsFindMin = false;											// 是否找到底端
	float MinData;
	int offsetIndex = sizeIndex * size * cols + rowIndex * cols;
	int CurrentChooseN = 0;
	float lastData = -1;
	for (int i = startIndex; i < cols; i++)
	{
		if (PointType[i + offsetIndex] == 2)
		{
			// 如果前面已經有找到其他點的話
			if (IsFindMin)
			{
				if (lastData != -1)
				{
					CurrentChooseN++;

					// 代表已經找滿了
					if (CurrentChooseN >= ChooseBestN)
						break;
				}
				lastData = -1;
			}

			IsFindMin = true;
			MinData = DataArray[i + offsetIndex];
		}
		else if (
			IsFindMin &&											// 要先找到最低點
			DataArray[i + offsetIndex] - MinData > Threshold &&		// 接著找大於這個 Threshold
			lastData == -1
			)
		{
			lastData = DataArray[i + offsetIndex] - MinData;
			PointType_BestN[sizeIndex * rows * ChooseBestN + rowIndex * ChooseBestN + CurrentChooseN] = i;
		}
	}

	// 接這著要判斷是否找到邊界
	// 如果沒有找到邊界，就回傳 -1
	//if (!IsFindBorder)
	for (int i = CurrentChooseN; i < ChooseBestN; i++)
		PointType_BestN[sizeIndex * rows * ChooseBestN + rowIndex * ChooseBestN + i] = -1;

}
__global__ static void ConnectPointsStatus(int* PointType_BestN, int* ConnectStatus, int size, int rows, int ChooseBestN, int ConnectRadius)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size * rows * ChooseBestN)						// 判斷是否超出大小
		return;

	// 算 Index
	int sizeIndex = id / (rows * ChooseBestN);
	int tempID = id % (rows * ChooseBestN);
	int rowIndex = tempID / ChooseBestN;
	int chooseIndex = tempID % ChooseBestN;

	// 代表這個點沒有有效的點
	if (PointType_BestN[sizeIndex * rows * ChooseBestN + rowIndex * ChooseBestN + chooseIndex] == -1)
		return;

	// 如果是有效的點，就繼續往下追 
	int finalPos = min(rowIndex + ConnectRadius, rows);		// 截止條件
	for (int i = rowIndex + 1; i < finalPos; i++)
	{
		for (int j = 0; j < ChooseBestN; j++)
		{
			// 下一個點的位置 (第 i 個 row 的點)
			// 然後的第 1 個點
			if (PointType_BestN[sizeIndex * rows * ChooseBestN + i * ChooseBestN + j] != -1)
			{
				// 前面項為現在這個點
				// 後面項為往下的點
				int diffX =	PointType_BestN[sizeIndex * rows * ChooseBestN + rowIndex * ChooseBestN	+ chooseIndex] -
							PointType_BestN[sizeIndex * rows * ChooseBestN + i * ChooseBestN		+ j];
				int diffY = i - rowIndex;
				int Radius = diffX * diffX + diffY * diffY;

				// 0 沒有用到喔
				if (Radius < ConnectRadius * ConnectRadius)
				{
					// 張數的位移 + Row 的位移 + 現在在 Top N 的點 + 半徑的位移 + 往下 Top N 的結果
					int index = sizeIndex * rows * ChooseBestN * ConnectRadius * ChooseBestN +			// 張數
								rowIndex * ChooseBestN * ConnectRadius * ChooseBestN +					// Row
								chooseIndex * ConnectRadius * ChooseBestN +								// 現在在 Top N 
								 (i - rowIndex) * ChooseBestN +											// 半徑
								j;
					ConnectStatus[index] = Radius;
				}
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

	//PointType_BestN = new int[size * rows * ChooseBestN];
	//memset(PointType_BestN, 0, sizeof(int) * size * rows * ChooseBestN);
	//PointType_1D_2 = new int[size * rows];
	//memset(PointType_1D, 0, sizeof(int) * size * rows);
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
	int *GPU_PointType_BestN, *PointType_BestN;
	cudaMalloc(&GPU_PointType_BestN, sizeof(int) * size * rows * ChooseBestN);
	PickBestChoiceToArray << <NumBlocks_small, NumThreads_small >> > (GPU_VolumeData_1D, GPU_PointType, GPU_PointType_BestN, size, rows, cols, ChooseBestN, StartIndex, GoThroughThreshold);
	CheckCudaError();

	// 連結點
	// 這個的大小 為 => 張數 * 250(rows) * 取幾個最大值(ChooseBestN個) * 每個最大值底下有 半徑個 (Raidus)  * 的下 N 排的幾個最大值(ChooseBestN) 
	int *GPU_Connect_Status;
	int ConnectStateSize = size * rows * ChooseBestN * ConnectRadius * ChooseBestN;
	cudaMalloc(&GPU_Connect_Status, sizeof(int) * ConnectStateSize);
	cudaMemset(GPU_Connect_Status, 0, sizeof(int) * ConnectStateSize);
	ConnectPointsStatus << <NumBlocks_small, NumThreads_small >> > (GPU_PointType_BestN, GPU_Connect_Status, size, rows, ChooseBestN, ConnectRadius);
	CheckCudaError();


	// 把資料傳回 CPU
	int *Connect_Status = new int[ConnectStateSize];
	PointType_BestN = new int[size * rows * ChooseBestN];
	cudaMemcpy(PointType,		GPU_PointType,		sizeof(uchar) * size * rows * cols,				cudaMemcpyDeviceToHost);
	cudaMemcpy(Connect_Status,	GPU_Connect_Status, sizeof(int) * ConnectStateSize,					cudaMemcpyDeviceToHost);
	cudaMemcpy(PointType_BestN, GPU_PointType_BestN, sizeof(int) * size * rows * ChooseBestN,		cudaMemcpyDeviceToHost);
	CheckCudaError();

	GetLargeLine(PointType_BestN, Connect_Status);
	#pragma endregion
	#pragma region Free Memory
	cudaFree(GPU_VolumeData_1D);
	cudaFree(GPU_PointType);
	cudaFree(GPU_PointType_BestN);
	cudaFree(GPU_Connect_Status);

	delete[] Connect_Status;
	delete[] PointType_BestN;
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
	//clock_t time;
	//time = clock();
	#pragma endregion
	#pragma region 透過 GPU 平行轉值
	// 原 Data Array
	float* GPU_VolumeData_1D;
	cudaMalloc(&GPU_VolumeData_1D, sizeof(float) * size * rows * cols);
	cudaMemcpy(GPU_VolumeData_1D, VolumeData_1D, sizeof(float) * size * rows * cols, cudaMemcpyHostToDevice);
	CheckCudaError();

	// Output Uint Array
	// 圖片的資料
	uchar *GPU_UintDataArray, *UintDataArray;
	cudaMalloc(&GPU_UintDataArray, sizeof(uchar) * size * rows * cols);
	CheckCudaError();

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
		/*Vec3b color[3];
		color[0] = Vec3b(0, 255, 255);
		color[1] = Vec3b(255, 255, 0);
		color[2] = Vec3b(255, 255, 255);
		for (int i = 0; i < size; i++)
			for (int j = 0; j < rows; j++)
			{
				for (int k = 0; k < ChooseBestN; k++)
				{
					int index = i * rows * ChooseBestN + j * ChooseBestN + k;
					if (PointType_BestN[index] != -1)
					{
						Point contourPoint(PointType_BestN[index], j);
						circle(ImgArray[i], contourPoint, 2, color[k], CV_FILLED);
					}
				}
			}*/
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

//////////////////////////////////////////////////////////////////////////
// 處理相關
//////////////////////////////////////////////////////////////////////////
void CudaBorder::GetMinMaxValue(float* begin, float& max, int size)
{
	clock_t time = clock();
	float* iter = max_element(begin, begin + size);
	unsigned int position = iter - begin;
	float max_val = *iter;
	//cout << "最大值是: " << max_val << " 在位置: " << position << endl;
	max = max_val;
}
void CudaBorder::GetLargeLine(int *PointType_BestN, int *Connect_Status)
{
	clock_t time = clock();

	// 選 N 個
	#pragma omp parallel for //num_thread(4)
	for (int i = 0; i < size; i++)
	{
		// 每個 10 段下去 Sample
		int RowGap = rows / 10;
		vector<vector<ConnectInfo>> StatusVector;
		
		for (int j = 0; j < rows; j += RowGap)
			for (int chooseNIndex = 0; chooseNIndex < ChooseBestN; chooseNIndex++)
			{
				int begin = j;
				int end = j;

				// 代表這個點沒有東西，所以略過
				if (PointType_BestN[i * rows * ChooseBestN + j * ChooseBestN + chooseNIndex] == -1)
					continue;

				// 連接狀況
				vector<ConnectInfo> Connect;

				#pragma region 往上找
				// 先加上自己
				ConnectInfo info;
				info.rowIndex = j;
				info.chooseIndex = chooseNIndex;
				Connect.push_back(info);

				int FindIndex = j;
				int FindChooseIndex = chooseNIndex;
				bool IsFind = true;
				while (IsFind && FindIndex > 0)
				{
					int minMoveIndex = -1;
					int minChooseIndex = -1;
					int tempValue = ConnectRadius * ConnectRadius;
					for (int k = 1; k < ConnectRadius; k++)
						for (int nextChooseNIndex = 0; nextChooseNIndex < ChooseBestN; nextChooseNIndex++)
						{
							int index = i * rows * ChooseBestN * ConnectRadius * ChooseBestN +					// Size
										(FindIndex - k) * ChooseBestN * ConnectRadius * ChooseBestN +			// Rows
										nextChooseNIndex * ConnectRadius * ChooseBestN +						// 現在在的 Top N 的點 (這邊要注意，這邊應該要放的是 要找的那個點的 ChooseIndex)
										k * ChooseBestN +														// 半徑
										FindChooseIndex;
							if (FindIndex - k >= 0 && Connect_Status[index] != 0 && tempValue > Connect_Status[index])
							{
								tempValue = Connect_Status[index];
								minMoveIndex = k;
								minChooseIndex = nextChooseNIndex;
							}
						}

					// 判斷是否有找到，找到就繼續找
					if (minMoveIndex != -1)
					{
						// 更便位置
						FindIndex = FindIndex - minMoveIndex;
						FindChooseIndex = minChooseIndex;

						// 丟進陣列
						info.rowIndex = FindIndex;
						info.chooseIndex = minChooseIndex;
						Connect.push_back(info);

						// 有找到
						IsFind = true;
					}
					else
						IsFind = false;
				}
				#pragma endregion
				#pragma region 往下找
				FindIndex = j;
				FindChooseIndex = chooseNIndex;
				while (IsFind && FindIndex < rows - 1)
				{
					int minMoveIndex = -1;
					int minChooseIndex = -1;				
					int tempValue = ConnectRadius * ConnectRadius;
					for (int k = 1; k < ConnectRadius; k++)
						for (int nextChooseNIndex = 0; nextChooseNIndex < ChooseBestN; nextChooseNIndex++)
						{
							int index = i * rows * ChooseBestN * ConnectRadius * ChooseBestN +					// Size
										FindIndex * ChooseBestN * ConnectRadius * ChooseBestN +					// Rows
										FindChooseIndex * ConnectRadius * ChooseBestN +							// 現在在的 Top N 的點
										k * ChooseBestN +														// 半徑
										nextChooseNIndex;
							if (FindIndex + k < rows && Connect_Status[index] != 0 && tempValue > Connect_Status[index])
							{
								tempValue = Connect_Status[index];
								minMoveIndex = k;
								minChooseIndex = nextChooseNIndex;
							}
						}

					// 判斷是否有找到，找到就繼續找
					if (minMoveIndex != -1)
					{
						// 更便位置
						FindIndex = FindIndex + minMoveIndex;
						FindChooseIndex = minChooseIndex;

						// 丟進陣列
						info.rowIndex = FindIndex;
						info.chooseIndex = minChooseIndex;
						Connect.push_back(info);

						// 有找到
						IsFind = true;
					}
					else
						IsFind = false;
				}
				#pragma endregion
			
				// 判斷是否有連出東西，如果連出來的東西大於 1
				if (Connect.size() > 1)
				{
					// 由小排到大
					sort(Connect.begin(), Connect.end(), SortByRows);
					StatusVector.push_back(Connect);
				}
			}
		

		// 前面的幾個張數，可能會找不到點，所以跳過處理
		if (StatusVector.size() == 0)
		{
			memset(&PointType_1D[i * rows], -1, sizeof(int) * rows);
			continue;
		}

		// 排序之後取最大
		sort(StatusVector.begin(), StatusVector.end(), SortByVectorSize);

		// 每個點續算變化率
		vector<float> DiffRateVector;
		for (int j = 0; j < StatusVector.size(); j++)
		{
			float diffRate = 0;
			for (int k = 0; k < StatusVector[j].size() - 1; k++)
			{
				int	FirstYIndex = i * rows * ChooseBestN +							// 張
								StatusVector[j][k].rowIndex * ChooseBestN +			// row
								StatusVector[j][k].chooseIndex;						// ChooseIndex
				int NextYIndex = i * rows * ChooseBestN +							// 張
								StatusVector[j][k + 1].rowIndex * ChooseBestN +		// row
								StatusVector[j][k + 1].chooseIndex;					// ChooseIndex
				float Diff = sqrt((float)(PointType_BestN[NextYIndex] - PointType_BestN[FirstYIndex]) * (PointType_BestN[NextYIndex] - PointType_BestN[FirstYIndex]));
				diffRate += Diff;
			}
			DiffRateVector.push_back(diffRate);
		}

		// 每一個結果去算變化量最大的
		int DiffMaxIndex = distance(DiffRateVector.begin(), max_element(DiffRateVector.begin(), DiffRateVector.end()));

		vector<ConnectInfo> LineVector = StatusVector[DiffMaxIndex];
		int index = 0;				// LineVector Index
		for (int j = 0; j < rows; j++)
		{
			int Type1D_Index = i * rows + j;
			if (LineVector[index].rowIndex != j)
				PointType_1D[Type1D_Index] = -1;
			else if (LineVector[index].rowIndex == j)
			{
				int BestN_Index = i * rows * ChooseBestN +							// 張
								LineVector[index].rowIndex * ChooseBestN +			// row
								LineVector[index].chooseIndex;						// ChooseIndex

				// 放進 PointType
				PointType_1D[j + i * rows] = PointType_BestN[BestN_Index];
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

	time = clock() - time;
	cout << "連接呈現: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
}
bool CudaBorder::SortByRows(ConnectInfo left, ConnectInfo right)
{
	return left.rowIndex < right.rowIndex;
}
bool CudaBorder::SortByVectorSize(vector<ConnectInfo> left, vector<ConnectInfo> right)
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
