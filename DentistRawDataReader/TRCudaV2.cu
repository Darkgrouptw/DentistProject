﻿#include "TRCudaV2.cuh"
#include "EigenUtility.h"

TRCudaV2::TRCudaV2()
{
}
TRCudaV2::~TRCudaV2()
{
}

//////////////////////////////////////////////////////////////////////////
// GPU
//////////////////////////////////////////////////////////////////////////
__device__ static float Z1Function(float x1)
{
	// 這個 Function 不確定在幹嘛XD
	// https://i.imgur.com/QS3bczf.png
	return -126.4517 + 
		0.4005123 *				x1 -
		0.000011981 *			pow(x1 - 2122.41, 2) -
		0.000000011664 *		pow(x1 - 2122.41, 3) +
		0.000000000001432 *		pow(x1 - 2122.41, 4) -
		0.0000000000000008164 * pow(x1 - 2122.41, 5) +
		5.939E-20 *				pow(x1 - 2122.41, 6);
}
__global__ static void RawDataToOriginalData(char* FileRawData, int* OCTRawData, int OCTDataSize)
{
	// 這邊是原本讀取是 1個 Byte 要轉乘 2個 Bytes 為一筆資料
	int id = blockIdx.y * gridDim.x * gridDim.z * blockDim.x +			// Y	=> Y * 250 * (2 * 1024)
		blockIdx.x * gridDim.z * blockDim.x +							// X	=> X * (2 * 1024)
		blockIdx.z * blockDim.x +										// Z	=> (Z1 * 1024 + Z2)
		threadIdx.x;

	// 這邊應該是不會發生，就當作例外判斷
	if (id >= OCTDataSize)
	{
		printf("轉 Raw Data 有 Error!\n");
		return;
	}

	OCTRawData[id] = (int)((uchar)FileRawData[id * 2] + (uchar)FileRawData[id * 2 + 1] * 256);
}
__global__ static void CombineTwoChannels(int* OCTData_2Channls, int* OCTData, int SizeX, int SizeZ)
{
	// 這邊是 Denoise，把兩個 Channel 的資料相加
	int id = blockIdx.y * gridDim.x * gridDim.z * blockDim.x +			// Y	=> Y * 250 * (2 * 1024)
		blockIdx.x * gridDim.z * blockDim.x +							// X	=> X * (2 * 1024)
		blockIdx.z * blockDim.x +										// Z	=> (Z1 * 1024 + Z2)
		threadIdx.x;

	int BoxSize = SizeX * SizeZ * 2;									// 一個 Channel 的資料是 正掃 + 反掃
	int BoxIndex = id / BoxSize;
	int BoxLeft = id % BoxSize;

	OCTData[id] = (OCTData_2Channls[BoxIndex * 2 * BoxSize + BoxLeft] +
				OCTData_2Channls[(BoxIndex * 2 + 1) * BoxSize + BoxLeft]) / 2;
}
__global__ static void ReverseBackScanData(int* OCTData, int SizeX, int SizeY, int SizeZ)
{
	// 這邊是要反轉 反掃的資料
	int id = (blockIdx.y * 2 + 1) * gridDim.x * 2 * gridDim.z * blockDim.x +			// Y	=> (Y * 2 + 1) * (2 * 1024)						=> 1, 3, 5, 7, 9
		blockIdx.x * gridDim.z * blockDim.x +											// X	=> X * (125 * 2) * (2 * 1024)
		blockIdx.z * blockDim.x +														// Z	=> (Z1 * 1024 + Z2)
		threadIdx.x;

	int changeID = (blockIdx.y * 2 + 1) * gridDim.x * 2 * gridDim.z * blockDim.x +		// Y	=> (Y * 2 + 1) * (2 * 1024)						=> 1, 3, 5, 7, 9
		(gridDim.y * 2 - blockIdx.x - 1) * gridDim.z * blockDim.x +						// X	=> (250 - X - 1) * (125 * 2) * (2 * 1024)
		blockIdx.z * blockDim.x +														// Z	=> (Z1 * 1024 + Z2)
		threadIdx.x;

	int value = OCTData[id];
	OCTData[id] = OCTData[changeID];
	OCTData[changeID] = value;
}
__global__ static void GetMatrixA(int* OCTData, float* MatrixA, int NumPolynomial, int OneDataSize)
{
	// 這個 Function 是去取得 MatrixA 的值
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// 例外判斷 (理論上應該也是不會超過)
	if (id >= (NumPolynomial + 1) * (NumPolynomial + 1))
	{
		printf("多項式 Fitting 有問題!\n");
		return;
	}

	// 算 Index
	int rowIndex = id % (NumPolynomial + 1);
	int colsIndex = id / (NumPolynomial + 1);

	// 做相加
	float value = 0;
	for (int i = 0; i < OneDataSize; i++)
	{
		// 抓出兩項的值
		float FirstValue = (float)i / OneDataSize;
		float SecondValue = (float)i / OneDataSize;
		value += pow(FirstValue, NumPolynomial - rowIndex) * pow(SecondValue, NumPolynomial - colsIndex);
	}
	MatrixA[id] = value;
}
__global__ static void GetMatrixB(int* OCTData, float* MatrixB, float YAverage, int NumPolynomial, int OneDataSize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// 算 Index
	int rowIndex = id % (NumPolynomial + 1);
	int colsIndex = id / (NumPolynomial + 1);

	// 做相加
	float value = 0;
	for (int i = 0; i < OneDataSize; i++)
	{
		// 抓出兩項的值
		float FirstValue = (float)i / OneDataSize;
		float SecondValue = OCTData[i] - YAverage;
		value += pow(FirstValue, NumPolynomial - rowIndex) * SecondValue;
	}
	MatrixB[id] = value;
}
__global__ static void MinusByFittingFunction(int* OCTData, float* PolyValue, int SizeZ)
{	
	// 這邊要減掉 Fitting Data
	int id = blockIdx.y * gridDim.x * gridDim.z * blockDim.x +			// Y	=> Y * 250 * (2 * 1024)
		blockIdx.x * gridDim.z * blockDim.x +							// X	=> X * (2 * 1024)
		blockIdx.z * blockDim.x +										// Z	=> (Z1 * 1024 + Z2)
		threadIdx.x;

	// 先拿出他是第幾個 Z
	int idZ = id % SizeZ;

	// 減掉預測的值
	OCTData[id] -= PolyValue[idZ];
}
__global__ static void ComputePXScale(float* PXScale, int OffsetBegin, int ShiftValue, int Steps, int Size)
{
	// 這邊是算出 PXScale Array(詳細在幹嘛我不是很懂@@)
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= Size)
	{
		printf("ComputePXScale 有問題!\n");
		return;
	}

	// 聽說是去直流
	int idOffset = OffsetBegin + ShiftValue;
	PXScale[id] = (Z1Function(idOffset + id) - Z1Function(idOffset)) * Steps;
}
__global__ static void FrequencyAdjust(int* OCTData, float* KSpaceData, float* PXScale, int* IndexArray, int CutIndex, int SizeX, int SizeY, int SizeZ)
{
	// 這邊是 Denoise，把兩個 Channel 的資料相加
	int id = blockIdx.y * gridDim.x * gridDim.z * blockDim.x +			// Y	=> Y * 250 * (2 * 1024)
		blockIdx.x * gridDim.z * blockDim.x +							// X	=> X * (2 * 1024)
		blockIdx.z * blockDim.x +										// Z	=> (Z1 * 1024 + Z2)
		threadIdx.x;

	if (id >= SizeX * SizeY * SizeZ)
	{
		printf("Frequency 轉換的地方有問題");
		return;
	}

	// 算回原本的 Index
	int idZ = id % SizeZ;
	if (IndexArray[idZ] == -1 || idZ >= CutIndex || idZ == 0)
	{
		KSpaceData[id] = 0;
		return;
	}

	// 要算斜率前，先拿出上一筆資料
	int LastPXScaleIndex = (IndexArray[idZ] - 1 <= 0 ? 0 : IndexArray[idZ] - 1);

	double m = (double)(OCTData[id] - OCTData[id - 1]) / (PXScale[IndexArray[idZ]] - PXScale[LastPXScaleIndex]);
	double c = OCTData[id] - m * PXScale[IndexArray[idZ]];
	KSpaceData[id] = m * idZ + c;
}
__global__ static void DataToComplexData(float* KSpaceData, cufftComplex* FFTData, int OCTDataSize)
{	
	// 把 KSpace 的 Data 塞進 FFT
	int id = blockIdx.y * gridDim.x * gridDim.z * blockDim.x +			// Y	=> Y * 250 * (2 * 1024)
		blockIdx.x * gridDim.z * blockDim.x +							// X	=> X * (2 * 1024)
		blockIdx.z * blockDim.x +										// Z	=> (Z1 * 1024 + Z2)
		threadIdx.x;

	if (id >= OCTDataSize)
	{
		printf("放進 Complex Data 有錯誤!!\n");
		return;
	}

	// 放進 Complex Data 裡
	FFTData[id].x = KSpaceData[id];
	FFTData[id].y = 0;
}
__global__ static void ComplexDataToData(cufftComplex* FFTData, float* OCTFloatData, int SizeX, int SizeY, int SizeZ, int OCTDataSize)
{
	// FFT 資料塞回原本的資料集
	int id = blockIdx.y * gridDim.x * gridDim.z * blockDim.x +			// Y	=> Y * 250 * (2 * 1024)
		blockIdx.x * gridDim.z * blockDim.x +							// X	=> X * (1 * 1024)
		blockIdx.z * blockDim.x +										// Z	=> (0 * 1024 + Z2)
		threadIdx.x;

	if (id >= OCTDataSize / 2)
	{
		printf("Complex To Data 有錯誤!!\n");
		return;
	}

	// 這邊要避免 0 頻率與 最大頻率(由於只取一半的右邊，所以只拿 1024)，詳情請看 Youtube 連結 (你看學長有多好，都找連結給你了，還不看!!)
	// 這邊要除以 2 是因為它會對稱
	// 然後拿的順序要反過來 (由於東元那邊的程式是這樣)
	// 如果是最大頻率 (也就是 Size / 2 - 1 => 1023)，那就要去下一個 也就是 1022 
	/*int idZ = id % (SizeZ / 2);
	idZ = SizeZ / 2 - idZ - 1;
	if (idZ == SizeZ / 2 - 1)
		idZ--;*/
	int idZ = id % (SizeZ / 2);
	if (idZ == 0)
		idZ++;
	

	// 這邊的算法要對應回去原本的資料
	int tempIndex = id / (SizeZ / 2);
	int idX = tempIndex % SizeX;
	int idY = tempIndex / SizeX;
	int NewIndex = idY * SizeX * SizeZ + idX * SizeZ + idZ;
	float temp = sqrt(FFTData[NewIndex].x * FFTData[NewIndex].x + FFTData[NewIndex].y * FFTData[NewIndex].y);

	// 做一下例外判斷
	if (temp == 0)
		OCTFloatData[id] = 0;
	else
		OCTFloatData[id] = log10f(temp) * 10;
}
__global__ static void ShiftFinalData(float* AfterFFTData, float* ShiftData, int SizeX, int SizeY, int FinalSizeZ, int FinalDataSize)
{
	// 這邊要做位移
	// 由於硬體是這樣子 ↓
	// => | ->
	// ("->" 是指第一段，"=>" 是指第二段)
	int id = blockIdx.y * gridDim.x * gridDim.z * blockDim.x +			// Y	=> Y * 250 * (2 * 1024)
		blockIdx.x * gridDim.z * blockDim.x +							// X	=> X * (2 * 1024)
		blockIdx.z * blockDim.x +										// Z	=> (Z1 * 1024 + Z2)
		threadIdx.x;
	
	if (id >= FinalDataSize)
	{
		printf("Shift Data 有錯誤!!\n");
		return;
	}
	
	// 這邊的算法要對應回去原本的資料
	int idZ = id % FinalSizeZ;
	int tempIndex = id / FinalSizeZ;
	int idX = tempIndex % SizeX;
	int idY = tempIndex / SizeX;

	// SizeY 折回來
	// (0 ~ 124 125 ~ 249)
	//		↓
	// (125 ~ 249 0 ~ 124)
	idY = (idY + SizeY / 2) % SizeY;

	int NewIndex = idY * SizeX * FinalSizeZ + idX * FinalSizeZ + idZ;
	ShiftData[id] = AfterFFTData[NewIndex];
	//ShiftData[id] = AfterFFTData[id];
}
__global__ static void NormalizeData(float* ShiftData, float MaxValue, float MinValue, int FinalDataSize)
{
	// 這邊是根據資料的最大最小值，去做 Normalize 資料
	int id = blockIdx.y * gridDim.x * gridDim.z * blockDim.x +			// Y	=> Y * 250 * (2 * 1024)
		blockIdx.x * gridDim.z * blockDim.x +							// X	=> X * (2 * 1024)
		blockIdx.z * blockDim.x +										// Z	=> (Z1 * 1024 + Z2)
		threadIdx.x;

	// 例外判斷
	if (id >= FinalDataSize)
	{
		printf("Normaliza Data 超出範圍\n");
		return;
	}

	ShiftData[id] = (ShiftData[id] - MinValue) / (MaxValue - MinValue);
}

// 邊界部分
__global__ static void findMaxAndMinPeak(float* DataArray, uchar* PointType, int size, int rows, int cols, float MaxPeakThreshold)
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
__global__ static void PickBestChoiceToArray(float* DataArray, uchar* PointType, int* PointType_BestN, int size, int rows, int cols, int ChooseBestN, int startIndex, float Threshold)
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
				int diffX = PointType_BestN[sizeIndex * rows * ChooseBestN + rowIndex * ChooseBestN + chooseIndex] -
					PointType_BestN[sizeIndex * rows * ChooseBestN + i * ChooseBestN + j];
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

// 轉成圖片
__global__ static void TransforToImage(float* VolumeData_Normalized, uchar* ImageArray, int SizeX, int SizeY, int FinalSizeZ)
{
	// 這邊是將原本的資料，轉換完圖片
	int id = blockIdx.y * gridDim.x * gridDim.z * blockDim.x +			// Y	=> Y * 250 * 1 * 1024
		blockIdx.x * gridDim.z * blockDim.x +							// X	=> X * 1 * 1024
		blockIdx.z * blockDim.x +										// Z	=> (Z1 * 1024 + Z2)
		threadIdx.x;

	if (id >= SizeX * SizeY * FinalSizeZ)								// 判斷是否超出大小
		return;

	float data = VolumeData_Normalized[id] * 255;
	if (data >= 255)
		ImageArray[id] = 255;
	else if (data <= 0)
		ImageArray[id] = 0;
	else
		ImageArray[id] = (uchar)data;
}

//////////////////////////////////////////////////////////////////////////
// CPU
//////////////////////////////////////////////////////////////////////////
// 轉換 Function
void TRCudaV2::SingleRawDataToPointCloud(char* FileRawData, int DataSize, int SizeX, int SizeZ, long ShiftValue, double K_Step, int CutValue)
{
	cout << "這邊 目前沒有驗證塞兩個 Channel 的資料!!!!" << endl;
	// 算時間
	#ifdef SHOW_TRCUDAV2_TOTAL_TIME
	totalTime = clock();
	#endif

	//////////////////////////////////////////////////////////////////////////
	// 步驟說明
	// 1. 上傳 GPU Data
	// 2. 一開始要把資料讀進來 (由於原本的資料都是 2個 Bytes 為一組，但 QT 目前是先用 GPU 轉換到 2個 Bytes)，和
	//    由於資料有 兩個 Channels，要相加除以2，可以去除雜訊 (由於原本的能量強度資料是使用三角波，所以會有去跟回兩個資料，就是把這兩筆資料相加除以 2)
	// 3. 用 5 次項去 Fit 一條曲線
	// 4. λ Space 轉成 K Space
	// 5. cuFFT
	// (這個部分不用位移)
	// 7. 根據最大最小值來 Normalize 資料
	// 8. 轉成圖
	// 9. 邊界判斷
	// 10. 抓下 GPU Data
	//
	// 細節說明：
	// 1. 轉換 Function => X 快軸、Y 慢軸
	// 2. ShiftValue	=> TRIGGER DELAY位移(換FIBER，電線校正回來用的)
	// 3. K_Step		=> 深度(14.多mm對應 2.5的k step；可以考慮之後用2)(k step越大，z軸越深，但資料精細度越差；1~2.5)
	// 4. CutValue		=> OCT每個z軸，前面數據減去多少。原因是開頭的laser弱，干涉訊號不明顯，拿掉的資料會比較美。 (東元那邊的變數是 cuteValue XD)
	//////////////////////////////////////////////////////////////////////////
	#pragma region 1. 上傳 GPU Data
	// 初始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	clock_t time = clock();
	#endif

	// GPU Data
	char* GPU_FileRawData;			// => 從檔案讀進來的 Raw Data
	int *GPU_OCTRawData_2Channel;	// => 這個是 OCT 掃完全部的 Raw Data (2Channels，如果"只用到一個" Channel 那就不會用到這個陣列)
	int *GPU_OCTRawData;			// => 這個是實際 Denoise 的 Data (也就是 CH1 + CH2 的資料) (如果"只有一個" Channel，就只會用到這個陣列)
	float *GPU_OCTFloatData;		// => 這個會用在兩個地方，一個是 K Space 的資料，一個是 FFT 後的資料

	// 是否是 2 Channels
	bool UseTwoChannels = (DataSize / SizeX / SizeZ == 4);		// 2 Byte & 2 Channles

	// 原始資料
	cudaMalloc(&GPU_FileRawData, sizeof(char) * DataSize);
	cudaMemcpy(GPU_FileRawData, FileRawData, sizeof(char) * DataSize, cudaMemcpyHostToDevice);
	CheckCudaError();

	// 判對是否使用 2 Chanels
	int OCTDataSize = SizeX * SizeZ;
	if (UseTwoChannels)
		cudaMalloc(&GPU_OCTRawData_2Channel, sizeof(int) * OCTDataSize * 2);
	cudaMalloc(&GPU_OCTRawData, sizeof(int) * OCTDataSize);
	cudaMalloc(&GPU_OCTFloatData, sizeof(float) * OCTDataSize);

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "1. 上傳至 GPU: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion 
	#pragma region 2. 讀檔轉換
	//////////////////////////////////////////////////////////////////////////
	// 這邊的資料格式是這樣
	// ↗↘↗↘ 是一組 (↗代表掃描 0 ~ 250的一次資料)
	// 其中一個 ↗↘ 是一個三角波的資料
	// 但因為有兩個 channel 所以一組資料是 ↗↘↗↘
	//////////////////////////////////////////////////////////////////////////
	// 開始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock();
	#endif
	
	// 解出 2 Byte 的資料
	if (UseTwoChannels)
	{
		RawDataToOriginalData << < dim3(SizeX, 1, SizeZ / NumThreads * 2), NumThreads >> > (GPU_FileRawData, GPU_OCTRawData_2Channel, DataSize / 2);
		CheckCudaError();

		// 兩個 Channel 作 Denoise
		CombineTwoChannels << < dim3(SizeX, 1, SizeZ / NumThreads), NumThreads >> > (GPU_OCTRawData_2Channel, GPU_OCTRawData, SizeX, SizeZ);

		// 刪除
		cudaFree(GPU_OCTRawData_2Channel);
	}
	else
		RawDataToOriginalData << < dim3(SizeX, 1, SizeZ / NumThreads), NumThreads >> > (GPU_FileRawData, GPU_OCTRawData, DataSize / 2);
	CheckCudaError();

	// 刪除 FileRaw Data
	cudaFree(GPU_FileRawData);

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "2. 讀檔轉換: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion
	#pragma region 3. 用五次項去 Fitting
	// 開始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock();
	#endif
	
	// 初始化 Matrix
	float* GPU_MatrixA;
	float* GPU_MatrixB;
	cudaMalloc(&GPU_MatrixA, sizeof(float) * (NumPolynomial + 1) *(NumPolynomial + 1));
	cudaMalloc(&GPU_MatrixB, sizeof(float) * (NumPolynomial + 1));

	// 先算平均
	int* FirstSizeZData = new int[SizeZ];
	cudaMemcpy(FirstSizeZData, GPU_OCTRawData, sizeof(int) * SizeZ, cudaMemcpyDeviceToHost);
	float average = std::accumulate(FirstSizeZData, FirstSizeZData + SizeZ, 0.0) / SizeZ;
	delete[] FirstSizeZData;

	// 取得 Matrix
	GetMatrixA << <1, (NumPolynomial + 1) * (NumPolynomial + 1) >> > (GPU_OCTRawData, GPU_MatrixA, NumPolynomial, SizeZ);
	GetMatrixB << <1, NumPolynomial + 1 >> > (GPU_OCTRawData, GPU_MatrixB, average, NumPolynomial, SizeZ);
	CheckCudaError();

	float* MatrixA = new float[(NumPolynomial + 1) *(NumPolynomial + 1)];
	float* MatrixB = new float[(NumPolynomial + 1)];
	cudaMemcpy(MatrixA, GPU_MatrixA, sizeof(float) * (NumPolynomial + 1) *(NumPolynomial + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(MatrixB, GPU_MatrixB, sizeof(float) * (NumPolynomial + 1), cudaMemcpyDeviceToHost);

	// 解 Eigen 找 Fitting Function
	EigenUtility eigen;
	eigen.SetAverageValue(average);
	eigen.SolveByEigen(MatrixA, MatrixB, NumPolynomial);
	
	// 扣除那個 Function
	float* GPU_PolyValue;
	float* PolyValue = eigen.GetFunctionArray(SizeZ, average);
	cudaMalloc(&GPU_PolyValue, sizeof(float) * SizeZ);
	cudaMemcpy(GPU_PolyValue, PolyValue, sizeof(float) * SizeZ, cudaMemcpyHostToDevice);
	MinusByFittingFunction << < dim3(SizeX, 1, SizeZ / NumThreads), NumThreads >> > (GPU_OCTRawData, GPU_PolyValue, SizeZ);
	CheckCudaError();

	// 刪除多出來的
	cudaFree(GPU_MatrixA);
	cudaFree(GPU_MatrixB);
	cudaFree(GPU_PolyValue);
	delete[] MatrixA;
	delete[] MatrixB;
	delete[] PolyValue;

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "3. 多項式去 Fitting : " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion
	#pragma region 4. λ Space 轉成 K Space
	// 開始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock();
	#endif

	// 初始化
	float* PX_Scale = new float[SizeZ];
	int* KSpaceIndexArray = new int[SizeZ];
	float* GPU_PXScale;
	int* GPU_KSpaceIndexArray;
	cudaMalloc(&GPU_PXScale,			sizeof(float) * SizeZ);
	cudaMalloc(&GPU_KSpaceIndexArray,	sizeof(int) * SizeZ);

	// 設定一些系數
	int OffsetBegin = 800;

	// 算出 PXScale 的 Array
	ComputePXScale << <SizeZ / NumThreads, NumThreads >> > (GPU_PXScale, OffsetBegin, ShiftValue, K_Step, SizeZ);
	CheckCudaError();

	// 抓下來準備算 K Space Index (由於這邊如果使用 GPU 去做，會導致大部分的 thread 在等最大工作量的 thread，所以這裡 CPU 做會比較快)
	cudaMemcpy(PX_Scale, GPU_PXScale, sizeof(float) * SizeZ, cudaMemcpyDeviceToHost);

	// 算 K Space 的對應 Array
	int index = 1;
	int KSpaceOffset = PX_Scale[SizeZ - 1];
	for (int i = 0; i <= KSpaceOffset; i++)
	{
		while (i >= PX_Scale[index])
		{
			index++;
		}
		KSpaceIndexArray[i] = index;
	}
	for (int i = KSpaceOffset + 1; i < SizeZ; i++)
		KSpaceIndexArray[i] = -1;

	// 由於 K Space 不是線性關係，所以要從 KSpaceIndexArray，找 Index，再從左右兩個點中，內插出實際在這個 Index 的值
	cudaMemcpy(GPU_KSpaceIndexArray, KSpaceIndexArray, sizeof(int) * SizeZ, cudaMemcpyHostToDevice);
	FrequencyAdjust << <dim3(SizeX, 1, SizeZ / NumThreads), NumThreads >> > (GPU_OCTRawData, GPU_OCTFloatData, GPU_PXScale, GPU_KSpaceIndexArray, KSpaceOffset - CutValue, SizeX, 1, SizeZ);
	CheckCudaError();

	// 釋放記憶體
	cudaFree(GPU_PXScale);
	cudaFree(GPU_KSpaceIndexArray);
	cudaFree(GPU_OCTRawData);
	delete[] KSpaceIndexArray;
	
	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "4. λ Space 轉成 K Space : " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion
	#pragma region 5. cuFFT
	// 開始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock();
	#endif

	cufftHandle PlanHandle;
	cufftComplex* GPU_ComplexData;

	// 這邊是創建 FFT 的 Handle & C2C 的 cufftComplex
	int NX = SizeZ;
	int BatchSize = SizeX;
	cufftPlan1d(&PlanHandle, NX, CUFFT_C2C, BatchSize);
	cudaMalloc(&GPU_ComplexData, sizeof(cufftComplex) * NX * BatchSize);
	CheckCudaError();	

	// 把資料塞進 Complex Data 裡
	//gpuDataToComplex << <512, 4 >> > (GPU_OCTFloatData, GPU_ComplexData, NX * BatchSize, 0);
	DataToComplexData << <dim3(SizeX, 1, SizeZ / NumThreads), NumThreads >> > (GPU_OCTFloatData, GPU_ComplexData, OCTDataSize);
	CheckCudaError();

	// 執行 cuFFT(CUDA™ Fast Fourier Transform) 
	cufftExecC2C(PlanHandle, GPU_ComplexData, GPU_ComplexData, CUFFT_FORWARD);
	CheckCudaError();

	// 刪除鏡向(FFT轉完之後會兩邊對稱) & 搬移資料
	// 想知道更多：https://www.youtube.com/watch?v=spUNpyF58BY
	//gpuComplexToData << <512, 4 >> > (GPU_ComplexData, GPU_OCTFloatData, NX * BatchSize / 2, SizeZ, 0);
	ComplexDataToData << <dim3(SizeX, 1, SizeZ / NumThreads / 2), NumThreads >> > (GPU_ComplexData, GPU_OCTFloatData, SizeX, 1, SizeZ, OCTDataSize);
	CheckCudaError();

	// 刪除
	cufftDestroy(PlanHandle);
	cudaFree(GPU_ComplexData);

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "5. cuFFT: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion
	#pragma region 7. Normaliza Data
	// 開始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock();
	#endif

	float *GPU_MaxElement = thrust::max_element(thrust::device, GPU_OCTFloatData, GPU_OCTFloatData + OCTDataSize / 2);
	float *GPU_MinElement = thrust::min_element(thrust::device, GPU_OCTFloatData, GPU_OCTFloatData + OCTDataSize / 2);

	// 抓下數值
	float MaxValue = 0;
	float MinValue = 0;
	cudaMemcpy(&MaxValue, GPU_MaxElement, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&MinValue, GPU_MinElement, sizeof(float), cudaMemcpyDeviceToHost);


	// 因為 Normalize Data 要做一件事情是  除 (Max - Min) ，要預防他除以 0
	// 所以這邊先判斷兩個是不是位置一樣 (因為如果整個 array 值都一樣，Min & Max 給的位置都會一樣(以驗證過)) (在抓下來之後，可以直接用直來判斷)
	assert(MaxValue != MinValue && "FFT後最大最小值一樣，資料有錯誤!!");
	NormalizeData << <dim3(SizeX, 1, SizeZ / NumThreads / 2), NumThreads >> > (GPU_OCTFloatData, MaxValue, MinValue, OCTDataSize / 2);

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "7. Normalize Data: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion
	#pragma region 8. 轉成圖
	// 開始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock();
	#endif

	// 圖片的資料
	uchar *GPU_UintDataArray;
	cudaMalloc(&GPU_UintDataArray, sizeof(uchar) * SizeX * 1 * SizeZ);
	CheckCudaError();

	// 轉圖片
	TransforToImage << <dim3(SizeX, 1, SizeZ / NumThreads / 2), NumThreads >> > (GPU_OCTFloatData, GPU_UintDataArray, SizeX, 1, SizeZ / 2);
	CheckCudaError();

	// 刪除記憶體
	cudaFree(GPU_OCTFloatData);

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "8. 轉成圖: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion
	#pragma region 10. 抓下 GPU Data
	// 開始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock();
	#endif
	
	// 刪除之前的資料
	SaveDelete(VolumeData);
	VolumeData = new uchar[SizeX * 1 * SizeZ];
	cudaMemcpy(VolumeData, GPU_UintDataArray, sizeof(uchar) * SizeX * 1 * SizeZ / 2, cudaMemcpyDeviceToHost);

	// 刪除 GPU
 	cudaFree(GPU_UintDataArray);

	// 設定一下其他參數
	this->size = 1;
	this->rows = SizeX;
	this->cols = SizeZ / 2;

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "10. 抓下 GPU Data : " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion
	
	// 結算
	#ifdef SHOW_TRCUDAV2_TOTAL_TIME
	totalTime = clock() - totalTime;
	cout << "轉換單張點雲: " << ((float)totalTime) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
}
void TRCudaV2::RawDataToPointCloud(char* FileRawData, int DataSize, int SizeX, int SizeY, int SizeZ, long ShiftValue, double K_Step, int CutValue)
{
	// 計算時間
	#ifdef SHOW_TRCUDAV2_TOTAL_TIME
	totalTime = clock();
	#endif

	//////////////////////////////////////////////////////////////////////////
	// 步驟說明
	// 1. 上傳 GPU Data
	// 2. 一開始要把資料讀進來 (由於原本的資料都是 2個 Bytes 為一組，但 QT 目前是先用 GPU 轉換到 2個 Bytes)，和
	//    由於資料有 兩個 Channels，要相加除以2，可以去除雜訊 (由於原本的能量強度資料是使用三角波，所以會有去跟回兩個資料，就是把這兩筆資料相加除以 2)
	// 3. 用 5 次項去 Fit 一條曲線
	// 4. λ Space 轉成 K Space
	// 5. cuFFT
	// 6. 位移 Data
	// 7. 根據最大最小值來 Normalize 資料
	// 8. 轉成圖
	// 9. 邊界判斷
	// 10. 抓下 GPU Data
	//
	// 細節說明：
	// 1. 轉換 Function => X 快軸、Y 慢軸
	// 2. ShiftValue	=> TRIGGER DELAY位移(換FIBER，電線校正回來用的)
	// 3. K_Step		=> 深度(14.多mm對應 2.5的k step；可以考慮之後用2)(k step越大，z軸越深，但資料精細度越差；1~2.5)
	// 4. CutValue		=> OCT每個z軸，前面數據減去多少。原因是開頭的laser弱，干涉訊號不明顯，拿掉的資料會比較美。 (東元那邊的變數是 cuteValue XD)
	// 5. 只是這邊比上方的 Function 多了 SizeY 個
	//////////////////////////////////////////////////////////////////////////
	#pragma region 1. 上傳 GPU Data
	// 初始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	clock_t time = clock();
	#endif

	// GPU Data
	char* GPU_FileRawData;			// => 從檔案讀進來的 Raw Data
	int *GPU_OCTRawData_2Channel;	// => 這個是 OCT 掃完全部的 Raw Data (2Channels，如果"只用到一個" Channel 那就不會用到這個陣列)
	int *GPU_OCTRawData;			// => 這個是實際 Denoise 的 Data (也就是 CH1 + CH2 的資料) (如果"只有一個" Channel，就只會用到這個陣列)
	float *GPU_OCTFloatData;		// => 這個會用在兩個地方，一個是 K Space 的資料，一個是 FFT 後的資料

	// 是否是 2 Channels
	bool UseTwoChannels = (DataSize / SizeX / SizeY / SizeZ == 4);	// 2 Byte & 2 Channles

	// 原始資料
	cudaMalloc(&GPU_FileRawData, sizeof(char) * DataSize);
	cudaMemcpy(GPU_FileRawData, FileRawData, sizeof(char) * DataSize, cudaMemcpyHostToDevice);
	CheckCudaError();

	// 判對是否使用 2 Chanels
	int OCTDataSize = SizeX * SizeY * SizeZ;
	if (UseTwoChannels)
		cudaMalloc(&GPU_OCTRawData_2Channel, sizeof(int) * OCTDataSize * 2);
	cudaMalloc(&GPU_OCTRawData, sizeof(int) * OCTDataSize);
	cudaMalloc(&GPU_OCTFloatData, sizeof(float) * OCTDataSize);

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "1. 上傳至 GPU: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion 
	#pragma region 2. 讀檔轉換
	//////////////////////////////////////////////////////////////////////////
	// 這邊的資料格式是這樣
	// ↗↘↗↘ 是一組 (↗代表掃描 0 ~ 250的一次資料)
	// 其中一個 ↗↘ 是一個三角波的資料
	// 但因為有兩個 channel 所以一組資料是 ↗↘↗↘
	//////////////////////////////////////////////////////////////////////////
	// 初始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock();
	#endif
	
	// 解出 2 Byte 的資料
	if (UseTwoChannels)
	{
		RawDataToOriginalData << < dim3(SizeX, SizeY, SizeZ / NumThreads * 2), NumThreads >> > (GPU_FileRawData, GPU_OCTRawData_2Channel, DataSize / 2);
		CheckCudaError();

		// 兩個 Channel 作 Denoise
		CombineTwoChannels << < dim3(SizeX, SizeY, SizeZ / NumThreads), NumThreads >> > (GPU_OCTRawData_2Channel, GPU_OCTRawData, SizeX, SizeZ);

		// 刪除
		cudaFree(GPU_OCTRawData_2Channel);
	}
	else
		RawDataToOriginalData << < dim3(SizeX, SizeY, SizeZ / NumThreads), NumThreads >> > (GPU_FileRawData, GPU_OCTRawData, DataSize / 2);
	CheckCudaError();

	// 反掃的資料，Index 要反轉
	ReverseBackScanData << < dim3(SizeX / 2, SizeY / 2, SizeZ / NumThreads), NumThreads >> > (GPU_OCTRawData, SizeX, SizeY, SizeZ);

	// 刪除 FileRaw Data
	cudaFree(GPU_FileRawData);

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "2. 讀檔轉換: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion
	#pragma region 3. 用五次項去 Fitting
	// 開始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock();
	#endif
	
	// 初始化 Matrix
	float* GPU_MatrixA;
	float* GPU_MatrixB;
	cudaMalloc(&GPU_MatrixA, sizeof(float) * (NumPolynomial + 1) *(NumPolynomial + 1));
	cudaMalloc(&GPU_MatrixB, sizeof(float) * (NumPolynomial + 1));

	// 先算平均
	int* FirstSizeZData = new int[SizeZ];
	cudaMemcpy(FirstSizeZData, GPU_OCTRawData, sizeof(int) * SizeZ, cudaMemcpyDeviceToHost);
	float average = std::accumulate(FirstSizeZData, FirstSizeZData + SizeZ, 0.0) / SizeZ;
	delete[] FirstSizeZData;

	// 取得 Matrix
	GetMatrixA << <1, (NumPolynomial + 1) * (NumPolynomial + 1) >> > (GPU_OCTRawData, GPU_MatrixA, NumPolynomial, SizeZ);
	GetMatrixB << <1, NumPolynomial + 1 >> > (GPU_OCTRawData, GPU_MatrixB, average, NumPolynomial, SizeZ);
	CheckCudaError();

	float* MatrixA = new float[(NumPolynomial + 1) *(NumPolynomial + 1)];
	float* MatrixB = new float[(NumPolynomial + 1)];
	cudaMemcpy(MatrixA, GPU_MatrixA, sizeof(float) * (NumPolynomial + 1) *(NumPolynomial + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(MatrixB, GPU_MatrixB, sizeof(float) * (NumPolynomial + 1), cudaMemcpyDeviceToHost);

	// 解 Eigen 找 Fitting Function
	EigenUtility eigen;
	eigen.SetAverageValue(average);
	eigen.SolveByEigen(MatrixA, MatrixB, NumPolynomial);
	
	// 扣除那個 Function
	float* GPU_PolyValue;
	float* PolyValue = eigen.GetFunctionArray(SizeZ, average);
	cudaMalloc(&GPU_PolyValue, sizeof(float) * SizeZ);
	cudaMemcpy(GPU_PolyValue, PolyValue, sizeof(float) * SizeZ, cudaMemcpyHostToDevice);
	MinusByFittingFunction << < dim3(SizeX, SizeY, SizeZ / NumThreads), NumThreads >> > (GPU_OCTRawData, GPU_PolyValue, SizeZ);
	CheckCudaError();

	// 刪除多出來的
	cudaFree(GPU_MatrixA);
	cudaFree(GPU_MatrixB);
	cudaFree(GPU_PolyValue);
	delete[] MatrixA;
	delete[] MatrixB;
	delete[] PolyValue;

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "3. 多項式去 Fitting : " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion
	#pragma region 4. λ Space 轉成 K Space
	// 開始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock();
	#endif

	// 初始化
	float* PX_Scale = new float[SizeZ];
	int* KSpaceIndexArray = new int[SizeZ];
	float* GPU_PXScale;
	int* GPU_KSpaceIndexArray;
	cudaMalloc(&GPU_PXScale,			sizeof(float) * SizeZ);
	cudaMalloc(&GPU_KSpaceIndexArray,	sizeof(int) * SizeZ);

	// 設定一些系數
	int OffsetBegin = 800;

	// 算出 PXScale 的 Array
	ComputePXScale << <SizeZ / NumThreads, NumThreads >> > (GPU_PXScale, OffsetBegin, ShiftValue, K_Step, SizeZ);
	CheckCudaError();

	// 抓下來準備算 K Space Index (由於這邊如果使用 GPU 去做，會導致大部分的 thread 在等最大工作量的 thread，所以這裡 CPU 做會比較快)
	cudaMemcpy(PX_Scale, GPU_PXScale, sizeof(float) * SizeZ, cudaMemcpyDeviceToHost);

	// 算 K Space 的對應 Array
	int index = 1;
	int KSpaceOffset = PX_Scale[SizeZ - 1];
	for (int i = 0; i <= KSpaceOffset; i++)
	{
		while (i >= PX_Scale[index])
		{
			index++;
		}
		KSpaceIndexArray[i] = index;
	}
	for (int i = KSpaceOffset + 1; i < SizeZ; i++)
		KSpaceIndexArray[i] = -1;

	// 由於 K Space 不是線性關係，所以要從 KSpaceIndexArray，找 Index，再從左右兩個點中，內插出實際在這個 Index 的值
	cudaMemcpy(GPU_KSpaceIndexArray, KSpaceIndexArray, sizeof(int) * SizeZ, cudaMemcpyHostToDevice);
	FrequencyAdjust << <dim3(SizeX, SizeY, SizeZ / NumThreads), NumThreads >> > (GPU_OCTRawData, GPU_OCTFloatData, GPU_PXScale, GPU_KSpaceIndexArray, KSpaceOffset - CutValue, SizeX, SizeY, SizeZ);
	CheckCudaError();

	// 釋放記憶體
	cudaFree(GPU_PXScale);
	cudaFree(GPU_KSpaceIndexArray);
	cudaFree(GPU_OCTRawData);
	delete[] KSpaceIndexArray;

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "4. λ Space 轉成 K Space : " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion
	#pragma region 5. cuFFT
	// 開始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock();
	#endif

	cufftHandle PlanHandle;
	cufftComplex* GPU_ComplexData;

	// 這邊是創建 FFT 的 Handle & C2C 的 cufftComplex
	int NX = SizeZ;
	int BatchSize = SizeX * SizeY;
	cufftPlan1d(&PlanHandle, NX, CUFFT_C2C, BatchSize);
	cudaMalloc(&GPU_ComplexData, sizeof(cufftComplex) * NX * BatchSize);
	CheckCudaError();	

	// 把資料塞進 Complex Data 裡
	DataToComplexData << <dim3(SizeX, SizeY, SizeZ / NumThreads), NumThreads >> > (GPU_OCTFloatData, GPU_ComplexData, OCTDataSize);
	CheckCudaError();

	// 執行 cuFFT(CUDA™ Fast Fourier Transform) 
	cufftExecC2C(PlanHandle, GPU_ComplexData, GPU_ComplexData, CUFFT_FORWARD);
	CheckCudaError();

	// 刪除鏡向(FFT轉完之後會兩邊對稱) & 搬移資料
	// 想知道更多：https://www.youtube.com/watch?v=spUNpyF58BY
	ComplexDataToData << <dim3(SizeX, SizeY, SizeZ / NumThreads / 2), NumThreads >> > (GPU_ComplexData, GPU_OCTFloatData, SizeX, SizeY, SizeZ, OCTDataSize);
	CheckCudaError();

	// 刪除
	cufftDestroy(PlanHandle);
	cudaFree(GPU_ComplexData);

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "5. cuFFT: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion
	#pragma region 6. 位移 Data
	// 開始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock();
	#endif

	float* GPU_ShiftData;
	cudaMalloc(&GPU_ShiftData, sizeof(float) * OCTDataSize / 2);		// 因為一半相同，所以去掉了
	
	// 這邊也是
	ShiftFinalData << <dim3(SizeX, SizeY, SizeZ / NumThreads / 2), NumThreads >> > (GPU_OCTFloatData, GPU_ShiftData, SizeX, SizeY, SizeZ / 2, OCTDataSize / 2);
	CheckCudaError();

	// 刪除記憶體
	cudaFree(GPU_OCTFloatData);

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "6. 搬移資料: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion
	#pragma region 7. Normaliza Data
	// 開始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock();
	#endif

	float *GPU_MaxElement = thrust::max_element(thrust::device, GPU_ShiftData, GPU_ShiftData + OCTDataSize / 2);
	float *GPU_MinElement = thrust::min_element(thrust::device, GPU_ShiftData, GPU_ShiftData + OCTDataSize / 2);

	// 抓下數值
	float MaxValue = 0;
	float MinValue = 0;
	cudaMemcpy(&MaxValue, GPU_MaxElement, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&MinValue, GPU_MinElement, sizeof(float), cudaMemcpyDeviceToHost);

	// 因為 Normaliza Data 要做一件事情是  除 (Max - Min) ，要預防他除以 0
	// 所以這邊先判斷兩個是不是位置一樣 (因為如果整個 array 值都一樣，Min & Max 給的位置都會一樣(以驗證過))
	assert(MaxValue != MinValue && "FFT後最大最小值一樣，資料有錯誤!!");
	NormalizeData << <dim3(SizeX, SizeY, SizeZ / NumThreads / 2), NumThreads >> > (GPU_ShiftData, MaxValue, MinValue, OCTDataSize / 2);

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "7. Normalize Data: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion
	#pragma region 8. 轉成圖
	// 開始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock();
	#endif

	// 圖片的資料
	uchar *GPU_UintDataArray;
	cudaMalloc(&GPU_UintDataArray, sizeof(uchar) * SizeX * SizeY * SizeZ);
	CheckCudaError();

	// 轉圖片
	TransforToImage << <dim3(SizeX, SizeY, SizeZ / NumThreads / 2), NumThreads >> > (GPU_ShiftData, GPU_UintDataArray, SizeX, SizeY, SizeZ / 2);
	CheckCudaError();

	// 設定一下其他參數
	size = SizeY;
	rows = SizeX;
	cols = SizeZ / 2;

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "8. 轉成圖: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion
	#pragma region 9. 抓取邊界
	// 開始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock();
	#endif

	#pragma region Init
	SaveDelete(PointType);
	PointType = new uchar[size * rows * cols];
	memset(PointType, 0, sizeof(uchar) * size * rows * cols);
	SaveDelete(PointType_1D);
	PointType_1D = new int[size * rows];
	memset(PointType_1D, 0, sizeof(int) * size * rows);

	// 點的型別
	uchar* GPU_PointType;
	cudaMalloc(&GPU_PointType, sizeof(uchar) * size * rows * cols);
	cudaMemset(GPU_PointType, 0, sizeof(uchar) * size * rows * cols);
	#pragma endregion
	#pragma region 抓取邊界
	// 找最大最小值
	findMaxAndMinPeak << < NumBlocks, NumThreads >> > (GPU_ShiftData, GPU_PointType, size, rows, cols, MaxPeakThreshold);
	CheckCudaError();

	// Parse 一些連續最小值
	ParseMaxMinPeak << < NumBlocks_small, NumThreads_small >> > (GPU_PointType, size, rows, cols, StartIndex);
	CheckCudaError();

	// 抓出一維陣列
	int *GPU_PointType_BestN, *PointType_BestN;
	cudaMalloc(&GPU_PointType_BestN, sizeof(int) * size * rows * ChooseBestN);
	PickBestChoiceToArray << <NumBlocks_small, NumThreads_small >> > (GPU_ShiftData, GPU_PointType, GPU_PointType_BestN, size, rows, cols, ChooseBestN, StartIndex, GoThroughThreshold);
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
	cudaMemcpy(PointType, GPU_PointType, sizeof(uchar) * size * rows * cols, cudaMemcpyDeviceToHost);
	cudaMemcpy(Connect_Status, GPU_Connect_Status, sizeof(int) * ConnectStateSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(PointType_BestN, GPU_PointType_BestN, sizeof(int) * size * rows * ChooseBestN, cudaMemcpyDeviceToHost);
	CheckCudaError();

	// 抓取最大的線
	GetLargeLine(PointType_BestN, Connect_Status);
	#pragma endregion

	// 刪除記憶體
	cudaFree(GPU_PointType);
	cudaFree(GPU_PointType_BestN);
	cudaFree(GPU_Connect_Status);
	cudaFree(GPU_ShiftData);

	delete[] Connect_Status;
	delete[] PointType_BestN;

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "9. 抓取邊界: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion
	#pragma region 10. 抓下 GPU Data
	// 開始
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock();
	#endif
	
	// 刪除之前的資料
	SaveDelete(VolumeData);
	VolumeData = new uchar[SizeX * SizeY * SizeZ / 2];
	cudaMemcpy(VolumeData, GPU_UintDataArray, sizeof(uchar) * SizeX * SizeY * SizeZ / 2, cudaMemcpyDeviceToHost);

	// 刪除 GPU
	cudaFree(GPU_UintDataArray);

	// 結算
	#ifdef SHOW_TRCUDAV2_DETAIL_TIME
	time = clock() - time;
	cout << "10. 抓下 GPU Data : " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	#pragma endregion

	// 結算
	#ifdef SHOW_TRCUDAV2_TOTAL_TIME
	totalTime = clock() - totalTime;
	cout << "轉換多張點雲: " << ((float)totalTime) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
}

// 拿出圖片
vector<Mat> TRCudaV2::TransfromMatArray(bool SaveBorder = false)
{
	// Total 時間
	#ifdef SHOW_TRCUDAV2_TOTAL_TIME
	totalTime = clock();
	#endif

	// 轉換到 Mat
	vector<Mat> ImgArray;
	for (int i = 0; i < size; i++)
	{
		// 根據 Offset 拿圖片
		Mat img(rows, cols, CV_8U, VolumeData + i * rows * cols);
		cvtColor(img, img, CV_GRAY2BGR);

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

	// 結算
	#ifdef SHOW_TRCUDAV2_TOTAL_TIME
	totalTime = clock() - totalTime;
	cout << "轉換圖片: " << ((float)totalTime) / CLOCKS_PER_SEC << " sec" << endl;
	#endif
	return ImgArray;
}

//////////////////////////////////////////////////////////////////////////
// Helper Function
//////////////////////////////////////////////////////////////////////////
void TRCudaV2::GetLargeLine(int *PointType_BestN, int *Connect_Status)
{
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
}
bool TRCudaV2::SortByRows(ConnectInfo left, ConnectInfo right)
{
	return left.rowIndex < right.rowIndex;
}
bool TRCudaV2::SortByVectorSize(vector<ConnectInfo> left, vector<ConnectInfo> right)
{
	return right.size() < left.size();
}
void TRCudaV2::CheckCudaError()
{
	cudaError GPU_Error = cudaGetLastError();
	if (GPU_Error != cudaSuccess)
	{
		cout << cudaGetErrorString(GPU_Error) << endl;
		assert(false);
		exit(-1);
	}
}
void TRCudaV2::SaveDelete(void* pointer)
{
	if (pointer != NULL)
		delete[] pointer;
}