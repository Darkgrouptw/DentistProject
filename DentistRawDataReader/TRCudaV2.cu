#include "TRCudaV2.cuh"
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
//__device__ static int GetScanIndex(int x, int y, int z)
//{
//	// 因為 Y % 2 == 0 的時候，順序是反過來的
//	// EX: 假設一次掃五個值，Index 為 
//	// 0, 1, 2, 3, 4 => 正著掃的 Index
//	// 9, 8, 7, 6, 5 => 反著掃的 Index
//	if (y % 2 == 0)
//	{
//	}
//}
__global__ static void RawDataToOriginalData(char *FileRawData, ushort *OCTRawData, int OCTDataSize)
{
	// 這邊是原本讀取是 1個 Byte 要轉乘 2個 Bytes 為一筆資料
	int id = blockIdx.x * gridDim.y * gridDim.z * blockDim.x +			// X => X * 250 * (2 * 1024)
		blockIdx.y * gridDim.z * blockDim.x +							// Y => Y * (2 * 1024)
		blockIdx.z * blockDim.x +										// Z1=> (Z1 * 1024 + Z2)
		threadIdx.x;													// Z2

	// 這邊應該是不會發生，就當作例外判斷
	if (id >= OCTDataSize)
	{
		printf("轉 Raw Data 有 Error!\n");
		return;
	}

	OCTRawData[id] = (uchar)FileRawData[id * 2] + (uchar)FileRawData[id * 2 + 1] * 256;
}
__global__ static void CombineTwoChannels(ushort *OCTData_2Channls, ushort *OCTData, int SizeX, int SizeZ)
{
	// 這邊是 Denoise，把兩個 Channel 的資料相加
	int id = blockIdx.x * gridDim.y * gridDim.z * blockDim.x +			// X	=> X * 250 * (2 * 1024)
		blockIdx.y * gridDim.z * blockDim.x +							// Y	=> Y * (2 * 1024)
		blockIdx.z * blockDim.x +										// Z1	=> (Z1 * 1024 + Z2)
		threadIdx.x;

	int BoxSize = SizeX * SizeZ * 2;									// 一個 Channel 的資料是 正掃 + 反掃
	int BoxIndex = id / BoxSize;
	int BoxLeft = id % BoxSize;

	OCTData[id] = (ushort)(((int)OCTData_2Channls[BoxIndex * 2 * BoxSize + BoxLeft] +
				(int)OCTData_2Channls[(BoxIndex * 2 + 1) * BoxSize + BoxLeft]) / 2);
}
__global__ static void ReverseBackScanData(ushort *OCTData, int SizeX, int SizeY, int SizeZ)
{
	// 這邊是要反轉 反掃的資料
	int id = (blockIdx.x * 2 + 1) * gridDim.y * 2 * gridDim.z * blockDim.x +			// X	=> (X * 2 + 1) * (125 * 2) * (2 * 1024)		=> 1, 3, 5, 6, 7
		blockIdx.y * gridDim.z * blockDim.x +											// Y	=> Y * (2 * 1024)	
		blockIdx.z * blockDim.x +														// Z1	=> (Z1 * 1024 + Z2)
		threadIdx.x;

	int changeID = (blockIdx.x * 2 + 1) * gridDim.y * 2 * gridDim.z * blockDim.x +		// X	=> (X * 2 + 1) * (125 * 2) * (2 * 1024)		=> 1, 3, 5, 6, 7
		(gridDim.y * 2 - blockIdx.y - 1) * gridDim.z * blockDim.x +							// Y	=> (250 - Y) * (2 * 1024)	
		blockIdx.z * blockDim.x +														// Z1	=> (Z1 * 1024 + Z2)
		threadIdx.x;

	ushort value = OCTData[id];
	OCTData[id] = OCTData[changeID];
	OCTData[changeID] = value;
}
__global__ static void GetMatrixA(ushort *OCTData, float *MatrixA, int NumPolynomial, int OneDataSize)
{
	// 這個 Function 是去取得 MatrixA 的值
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// 例外判斷 (理論上應該也是不會超過)
	if (id >= (NumPolynomial + 1) * (NumPolynomial +1 ))
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
__global__ static void GetMatrixB(ushort *OCTData, float *MatrixB, float Average, int NumPolynomial, int OneDataSize)
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
		float SecondValue = OCTData[i] - Average;
		value += pow(FirstValue, NumPolynomial - rowIndex) * SecondValue;
	}
	MatrixB[id] = value;
}
__global__ static void MinusByFittingFunction(ushort *OCTData, float* PolyParams, int NumPolynomial, int SizeX, int SizeY, int SizeZ)
{

}

//////////////////////////////////////////////////////////////////////////
// CPU
//////////////////////////////////////////////////////////////////////////
// 轉換 Function => X 快軸、Y 慢軸
void TRCudaV2::RawDataToPointCloud(char* FileRawData, int DataSize, int SizeX, int SizeY, int SizeZ)
{
	//////////////////////////////////////////////////////////////////////////
	// 步驟說明
	// 1. 上傳 GPU Data
	// 2. 一開始要把資料讀進來 (由於原本的資料都是 2個 Bytes 為一組，但 QT 目前是先用 GPU 轉換到 2個 Bytes)，和
	//    由於資料有 兩個 Channels，要相加除以2，可以去除雜訊 (由於原本的能量強度資料是使用三角波，所以會有去跟回兩個資料，就是把這兩筆資料相加除以 2)
	// 3. 用 5 次項去 Fit 一條曲線
	//
	//////////////////////////////////////////////////////////////////////////
	#pragma region 1. 上傳 GPU Data
	// 初始
	clock_t time = clock();

	// GPU Data
	// GPU_FileRawData				=> 從檔案讀進來的 Raw Data
	// GPU_OCTRawData_2Channel		=> 這個是 OCT 掃完全部的 Raw Data (2Channels，如果"只用到一個" Channel 那就不會用到這個陣列)
	// GPU_OCTRawData				=> 這個是實際 Denoise 的 Data (也就是 CH1 + CH2 的資料) (如果"只有一個" Channel，就只會用到這個陣列)
	char* GPU_FileRawData;
	ushort *GPU_OCTRawData_2Channel;
	ushort *GPU_OCTRawData;

	// 是否是 2 Channels
	bool UseTwoChannels = (DataSize / SizeX / SizeY / SizeZ == 4);	// 2 Byte & 2 Channles

	// 原始資料
	cudaMalloc(&GPU_FileRawData, sizeof(char) * DataSize);
	cudaMemcpy(GPU_FileRawData, FileRawData, sizeof(char) * DataSize, cudaMemcpyHostToDevice);
	CheckCudaError();

	// 判對是否使用 2 Chanels
	int OCTDataSize = SizeX * SizeY * SizeZ;
	if (UseTwoChannels)
		cudaMalloc(&GPU_OCTRawData_2Channel, sizeof(ushort) * OCTDataSize * 2);
	cudaMalloc(&GPU_OCTRawData, sizeof(ushort) * OCTDataSize);

	// 結算
	time = clock() - time;
	cout << "1. 上傳至 GPU: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#pragma endregion
	#pragma region 2. 讀檔轉換
	//////////////////////////////////////////////////////////////////////////
	// 這邊的資料格式是這樣
	// ↗↘↗↘ 是一組 (↗代表掃描 0 ~ 250的一次資料)
	// 其中一個 ↗↘ 是一個三角波的資料
	// 但因為有兩個 channel 所以一組資料是 ↗↘↗↘
	//////////////////////////////////////////////////////////////////////////

	// 初始
	time = clock();
	
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

	// 反掃的資料，Index 要反轉
	ReverseBackScanData << < dim3(SizeX / 2, SizeY / 2, SizeZ / NumThreads), NumThreads >> > (GPU_OCTRawData, SizeX, SizeY, SizeZ);

	// 結算
	time = clock() - time;
	cout << "2. 讀檔轉換: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#pragma endregion
	#pragma region 3. 用五次項去
	time = clock();
	
	// 初始化 Matrix
	float *GPU_MatrixA;
	float *GPU_MatrixB;
	cudaMalloc(&GPU_MatrixA, sizeof(float) * (NumPolynomial + 1) *(NumPolynomial + 1));
	cudaMalloc(&GPU_MatrixB, sizeof(float) * (NumPolynomial + 1));

	// 先算平均
	ushort *FirstSizeZData = new ushort[SizeZ];
	cudaMemcpy(FirstSizeZData, GPU_OCTRawData, sizeof(float) * SizeZ, cudaMemcpyDeviceToHost);
	float average = std::accumulate(FirstSizeZData, FirstSizeZData + SizeZ, 0.0) / SizeZ;
	delete[] FirstSizeZData;

	// 取得 Matrix
	GetMatrixA << <1, (NumPolynomial + 1) * (NumPolynomial + 1) >> > (GPU_OCTRawData, GPU_MatrixA, NumPolynomial, SizeZ);
	GetMatrixB << <1, NumPolynomial + 1 >> > (GPU_OCTRawData, GPU_MatrixB, average, NumPolynomial, SizeZ);
	CheckCudaError();

	float *MatrixA = new float[(NumPolynomial + 1) *(NumPolynomial + 1)];
	float *MatrixB = new float[(NumPolynomial + 1)];
	cudaMemcpy(MatrixA, GPU_MatrixA, sizeof(float) * (NumPolynomial + 1) *(NumPolynomial + 1)	, cudaMemcpyDeviceToHost);
	cudaMemcpy(MatrixB, GPU_MatrixB, sizeof(float) * (NumPolynomial + 1)						, cudaMemcpyDeviceToHost);

	// 解 Eigen 找 Fitting Function
	EigenUtility eigen;
	eigen.SetAverageValue(average);
	eigen.SolveByEigen(MatrixA, MatrixB, NumPolynomial);
	
	// 扣除那個 Function
	float *GPU_PolyParams;
	cudaMalloc(&GPU_PolyParams, sizeof(float) * (NumPolynomial + 1));
	cudaMemcpy(GPU_PolyParams, eigen.params, sizeof(float) * (NumPolynomial + 1), cudaMemcpyHostToDevice);
	MinusByFittingFunction << < dim3(SizeX, SizeY, SizeZ / NumThreads), NumThreads >> > (GPU_OCTRawData, GPU_PolyParams, NumThreads, SizeX, SizeY, SizeZ);

	// 刪除多出來的
	cudaFree(GPU_MatrixA);
	cudaFree(GPU_MatrixB);
	cudaFree(GPU_PolyParams);
	delete[] MatrixA;
	delete[] MatrixB;

	// 結算
	time = clock() - time;
	cout << "3. 多項式去 Fitting : " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#pragma endregion
	#pragma region 抓下 GPU Data
	// 刪除之前的資料
	SaveDelete(OCTData);
	OCTData = new ushort[SizeX * SizeY * SizeZ];

	cudaMemcpy(OCTData, GPU_OCTRawData, sizeof(ushort) * SizeX * SizeY * SizeZ, cudaMemcpyDeviceToHost);
	#pragma endregion
	#pragma region 刪除所有的變數
	cudaFree(GPU_FileRawData);
	cudaFree(GPU_OCTRawData);
	#pragma endregion
}

//////////////////////////////////////////////////////////////////////////
// Helper Function
//////////////////////////////////////////////////////////////////////////
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