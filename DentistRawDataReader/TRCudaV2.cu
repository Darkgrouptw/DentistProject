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
__device__ static double Z1Function(double x1)
{
	// 這個 Function 不確定在幹嘛XD
	// https://i.imgur.com/QS3bczf.png
	return -126.4517 + 0.4005123 * x1 -
		pow(0.000011981 * (x1 - 2122.41), 2) -
		pow(0.000000011664 * (x1 - 2122.41), 3) +
		pow(0.000000000001432 * (x1 - 2122.41), 4) -
		pow(0.0000000000000008164 * (x1 - 2122.41), 5) +
		pow(5.939E-20 * (x1 - 2122.41), 6);
}
__global__ static void RawDataToOriginalData(char *FileRawData, int *OCTRawData, int OCTDataSize)
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

	OCTRawData[id] = (int)((uchar)FileRawData[id * 2] + (uchar)FileRawData[id * 2 + 1] * 256);
}
__global__ static void CombineTwoChannels(int *OCTData_2Channls, int *OCTData, int SizeX, int SizeZ)
{
	// 這邊是 Denoise，把兩個 Channel 的資料相加
	int id = blockIdx.x * gridDim.y * gridDim.z * blockDim.x +			// X	=> X * 250 * (2 * 1024)
		blockIdx.y * gridDim.z * blockDim.x +							// Y	=> Y * (2 * 1024)
		blockIdx.z * blockDim.x +										// Z1	=> (Z1 * 1024 + Z2)
		threadIdx.x;

	int BoxSize = SizeX * SizeZ * 2;									// 一個 Channel 的資料是 正掃 + 反掃
	int BoxIndex = id / BoxSize;
	int BoxLeft = id % BoxSize;

	OCTData[id] = (OCTData_2Channls[BoxIndex * 2 * BoxSize + BoxLeft] +
				OCTData_2Channls[(BoxIndex * 2 + 1) * BoxSize + BoxLeft]) / 2;
}
__global__ static void ReverseBackScanData(int *OCTData, int SizeX, int SizeY, int SizeZ)
{
	// 這邊是要反轉 反掃的資料
	int id = (blockIdx.x * 2 + 1) * gridDim.y * 2 * gridDim.z * blockDim.x +			// X	=> (X * 2 + 1) * (125 * 2) * (2 * 1024)		=> 1, 3, 5, 6, 7
		blockIdx.y * gridDim.z * blockDim.x +											// Y	=> Y * (2 * 1024)	
		blockIdx.z * blockDim.x +														// Z1	=> (Z1 * 1024 + Z2)
		threadIdx.x;

	int changeID = (blockIdx.x * 2 + 1) * gridDim.y * 2 * gridDim.z * blockDim.x +		// X	=> (X * 2 + 1) * (125 * 2) * (2 * 1024)		=> 1, 3, 5, 6, 7
		(gridDim.y * 2 - blockIdx.y - 1) * gridDim.z * blockDim.x +						// Y	=> (250 - Y) * (2 * 1024)	
		blockIdx.z * blockDim.x +														// Z1	=> (Z1 * 1024 + Z2)
		threadIdx.x;

	int value = OCTData[id];
	OCTData[id] = OCTData[changeID];
	OCTData[changeID] = value;
}
__global__ static void GetMatrixA(int *OCTData, float *MatrixA, int NumPolynomial, int OneDataSize)
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
__global__ static void GetMatrixB(int *OCTData, float *MatrixB, float YAverage, int NumPolynomial, int OneDataSize)
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
__global__ static void MinusByFittingFunction(int *OCTData, float* PolyValue, int NumPolynomial, int SizeX, int SizeY, int SizeZ)
{	
	// 這邊要減掉 Fitting Data
	int id = blockIdx.x * gridDim.y * gridDim.z * blockDim.x +			// X => X * 250 * (2 * 1024)
		blockIdx.y * gridDim.z * blockDim.x +							// Y => Y * (2 * 1024)
		blockIdx.z * blockDim.x +										// Z1=> (Z1 * 1024 + Z2)
		threadIdx.x;													// Z2

	// 先拿出他是第幾個 Z
	int idZ = id % SizeZ;

	// 減掉預測的值
	OCTData[id] -= PolyValue[idZ];
}
__global__ static void ComputePXScale(double *PXScale, int OffsetBegin, int ShiftValue, int Steps, int Size)
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
__global__ static void FrequencyAdjust(int *OCTData, int *KSpaceData, double *PXScale, int* IndexArray, int CutValue, int SizeX, int SizeY, int SizeZ)
{
	// 這邊是 Denoise，把兩個 Channel 的資料相加
	int id = blockIdx.x * gridDim.y * gridDim.z * blockDim.x +			// X	=> X * 250 * (2 * 1024)
		blockIdx.y * gridDim.z * blockDim.x +							// Y	=> Y * (2 * 1024)
		blockIdx.z * blockDim.x +										// Z1	=> (Z1 * 1024 + Z2)
		threadIdx.x;

	if (id >= SizeX * SizeY * SizeZ)
	{
		printf("Frequency 轉換的地方有問題");
		return;
	}

	// 算回原本的 Index
	int idZ = id % SizeZ;
	if (IndexArray[idZ] == -1)
	{
		KSpaceData[id] = 0;
		return;
	}

	// 要算斜率前，先拿出上一筆資料
	int LastData = (idZ == 0 ? 0 : OCTData[id - 1]);
	int LastPXScaleIndex = (IndexArray[idZ] - 1 <= 0 ? 0 : IndexArray[idZ] - 1);

	double m = (double)(OCTData[id] - LastData) / (PXScale[IndexArray[idZ]] - PXScale[LastPXScaleIndex]);
	double c = OCTData[id] - m * PXScale[IndexArray[idZ]];
	KSpaceData[id] = m * idZ + c;
}

//////////////////////////////////////////////////////////////////////////
// CPU
//////////////////////////////////////////////////////////////////////////
void TRCudaV2::RawDataToPointCloud(char* FileRawData, int DataSize, int SizeX, int SizeY, int SizeZ, long ShiftValue, double K_Step, int CutValue)
{
	//////////////////////////////////////////////////////////////////////////
	// 步驟說明
	// 1. 上傳 GPU Data
	// 2. 一開始要把資料讀進來 (由於原本的資料都是 2個 Bytes 為一組，但 QT 目前是先用 GPU 轉換到 2個 Bytes)，和
	//    由於資料有 兩個 Channels，要相加除以2，可以去除雜訊 (由於原本的能量強度資料是使用三角波，所以會有去跟回兩個資料，就是把這兩筆資料相加除以 2)
	// 3. 用 5 次項去 Fit 一條曲線
	// 4. λ Space 轉成 K Space (好像跟硬體有關)
	// 5. FFT
	// 6. 126 & 1 的對調
	// 7. 抓下 GPU Data
	// 8. 刪除所有的變數
	//
	// 細節說明：
	// 1. 轉換 Function => X 快軸、Y 慢軸
	// 2. ShiftValue	=> TRIGGER DELAY位移(換FIBER，電線校正回來用的)
	// 3. K_Step		=> 深度(14.多mm對應 2.5的k step；可以考慮之後用2)(k step越大，z軸越深，但資料精細度越差；1~2.5)
	// 4. CutValue		=> OCT每個z軸，前面數據減去多少。原因是開頭的laser弱，干涉訊號不明顯，拿掉的資料會比較美。 (東元那邊的變數是 cuteValue XD)
	// 5. 要記得 Cuda 裡面的 blockIdx.x, blockIdx.y & blockIdx.y 跟 OCTMapData 實際上的 X, Y, Z是不一樣的喔 (實際上是 Z X Y，因為資料的讀取順序是 先讀 SizeZ 個 在讀 SizeX 在讀 SizeY)
	//////////////////////////////////////////////////////////////////////////
	#pragma region 1. 上傳 GPU Data
	// 初始
	clock_t time = clock();

	// GPU Data
	// GPU_FileRawData				=> 從檔案讀進來的 Raw Data
	// GPU_OCTRawData_2Channel		=> 這個是 OCT 掃完全部的 Raw Data (2Channels，如果"只用到一個" Channel 那就不會用到這個陣列)
	// GPU_OCTRawData				=> 這個是實際 Denoise 的 Data (也就是 CH1 + CH2 的資料) (如果"只有一個" Channel，就只會用到這個陣列)
	char* GPU_FileRawData;
	int *GPU_OCTRawData_2Channel;
	int *GPU_OCTRawData;
	int *GPU_KSpaceData;

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
	cudaMalloc(&GPU_KSpaceData, sizeof(int) * OCTDataSize);

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
	#pragma region 3. 用五次項去 Fitting
	// 開始
	time = clock();
	
	// 初始化 Matrix
	float *GPU_MatrixA;
	float *GPU_MatrixB;
	cudaMalloc(&GPU_MatrixA, sizeof(float) * (NumPolynomial + 1) *(NumPolynomial + 1));
	cudaMalloc(&GPU_MatrixB, sizeof(float) * (NumPolynomial + 1));

	// 先算平均
	int *FirstSizeZData = new int[SizeZ];
	cudaMemcpy(FirstSizeZData, GPU_OCTRawData, sizeof(int) * SizeZ, cudaMemcpyDeviceToHost);
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
	float *GPU_PolyValue;
	float *PolyValue = eigen.GetFunctionArray(SizeZ, average);
	cudaMalloc(&GPU_PolyValue, sizeof(float) * SizeZ);
	cudaMemcpy(GPU_PolyValue, PolyValue, sizeof(float) * SizeZ, cudaMemcpyHostToDevice);
	MinusByFittingFunction << < dim3(SizeX, SizeY, SizeZ / NumThreads), NumThreads >> > (GPU_OCTRawData, GPU_PolyValue, NumThreads, SizeX, SizeY, SizeZ);
	CheckCudaError();

	// 刪除多出來的
	cudaFree(GPU_MatrixA);
	cudaFree(GPU_MatrixB);
	cudaFree(GPU_PolyValue);
	delete[] MatrixA;
	delete[] MatrixB;
	delete[] PolyValue;

	// 結算
	time = clock() - time;
	cout << "3. 多項式去 Fitting : " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#pragma endregion
	#pragma region 4. λ Space 轉成 K Space (好像跟硬體有關)
	// 開始
	time = clock();

	// 初始化
	double *PX_Scale = new double[SizeZ];
	int *KSpaceIndexArray = new int[SizeZ];
	double *GPU_PXScale;
	int *GPU_KSpaceIndexArray;
	cudaMalloc(&GPU_PXScale,			sizeof(double) * SizeZ);
	cudaMalloc(&GPU_KSpaceIndexArray,	sizeof(int) * SizeZ);

	// 設定一些系數
	int OffsetBegin = 800;

	// 算出 PXScale 的 Array
	ComputePXScale << <SizeZ / NumThreads, NumThreads >> > (GPU_PXScale, OffsetBegin, ShiftValue, K_Step, SizeZ);
	CheckCudaError();

	// 抓下來準備算 K Space Index (由於這邊如果使用 GPU 去做，會導致大部分的 thread 在等最大工作量的 thread，所以這裡 CPU 做會比較快)
	cudaMemcpy(PX_Scale, GPU_PXScale, sizeof(double) * SizeZ, cudaMemcpyDeviceToHost);

	// 算 K Space 的對應 Array
	int index = 1;
	int KSpaceOffset = PX_Scale[SizeZ - 1];
	for (int i = 0; i <= KSpaceOffset; i++)
	{
		while (i > PX_Scale[index])
		{
			index++;
		}
		KSpaceIndexArray[i] = index;
	}
	for (int i = KSpaceOffset + 1; i < SizeZ; i++)
		KSpaceIndexArray[i] = -1;

	// 好像是要做轉
	cudaMemcpy(GPU_KSpaceIndexArray, KSpaceIndexArray, sizeof(int) * SizeZ, cudaMemcpyHostToDevice);
	FrequencyAdjust << <dim3(SizeX, SizeY, SizeZ / NumThreads), NumThreads >> > (GPU_OCTRawData, GPU_KSpaceData, GPU_PXScale, GPU_KSpaceIndexArray, index - CutValue, SizeX, SizeY, SizeZ);
	CheckCudaError();

	// 釋放記憶體
	cudaFree(GPU_PXScale);
	cudaFree(GPU_KSpaceIndexArray);
	delete[] KSpaceIndexArray;

	// 結算
	time = clock() - time;
	cout << "4. λ Space 轉成 K Space : " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#pragma endregion
	#pragma region 5. FFT
	//cufftHandle PlanHandle;
	//cufftComplex *ComplexData;

	//int NX = SizeZ;
	//int BatchSize = SizeX * SizeY;
	//cufftPlan1d(&PlanHandle, NX, CUFFT_C2C, BatchSize);
	//cudaMalloc(&ComplexData, sizeof(cufftComplex) * NX * BatchSize);

	// data in  gpuCutDataZ
	//gpuDataToComplex << <gridDimX, blockDimX >> > (gpuReversedData, gpuComplexData, NX * BATCH, NX * BATCH * i);
	//gpuError = cudaDeviceSynchronize();

	// excute cufft
	//cufftExecC2C(plan, gpuComplexData, gpuComplexData, CUFFT_FORWARD);
	//gpuError = cudaDeviceSynchronize();

	// data out and cut mirror data
	//gpuComplexToData << <gridDimX, blockDimX >> > (gpuComplexData, gpuFloatData, NX * BATCH / 2, size_Z, NX * BATCH / 2 * i);
	//gpuError = cudaDeviceSynchronize();

	//cufftDestroy(plan);
	//cudaFree(gpuComplexData);
	#pragma endregion
	#pragma region 6. 126 & 1 的對調

	#pragma endregion
	#pragma region 7. 抓下 GPU Data
	// 開始
	time = clock();
	
	// 刪除之前的資料
	SaveDelete(OCTData);
	OCTData = new int[OCTDataSize];

	//cudaMemcpy(OCTData, GPU_OCTRawData, sizeof(int) * OCTDataSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(OCTData, GPU_KSpaceData, sizeof(int) * OCTDataSize, cudaMemcpyDeviceToHost);

	// 結算
	time = clock() - time;
	cout << "7. 抓下 GPU Data : " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
	#pragma endregion
	#pragma region 8. 刪除所有的變數
	// 開始
	time = clock();

	cudaFree(GPU_FileRawData);
	cudaFree(GPU_OCTRawData);
	cudaFree(GPU_KSpaceData);

	// 結算
	time = clock() - time;
	cout << "8. 刪除 Data: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
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