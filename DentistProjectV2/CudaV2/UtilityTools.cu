#include "UtilityTools.cuh"
#include "EigenUtility.h"

UtilityTools::UtilityTools()
{
}
UtilityTools::~UtilityTools()
{
	SaveDelete(CPUImageData);
}

//////////////////////////////////////////////////////////////////////////
// GPU
//////////////////////////////////////////////////////////////////////////
__global__ static void CountEachCol(uchar3* GPUImageData, int4* CountEachImageColData, int SmoothSizeRange, int maxid)
{
	int id = threadIdx.x * gridDim.x * gridDim.y +
		blockIdx.x * gridDim.y +
		blockIdx.y;

	if (id >= maxid)return;

	int halfSize = (SmoothSizeRange - 1) / 2;

		for (int kk = -halfSize; kk <= halfSize; kk++)
		{
			if (((blockIdx.y + kk) < 0) || ((blockIdx.y + kk) >= gridDim.y))
				continue;

			int idx = threadIdx.x * gridDim.x * gridDim.y +
				blockIdx.x * gridDim.y +
				(blockIdx.y + kk);

			uchar3 color = GPUImageData[idx];

			// BGR
			if (color.x == 255 && color.y == 0 && color.z == 0)
				CountEachImageColData[id].x++;
			else if (color.x == 0 && color.y == 255 && color.z == 0)
				CountEachImageColData[id].y++;
			else if (color.x == 0 && color.y == 0 && color.z == 255)
				CountEachImageColData[id].z++;
			else if (color.x == 0 && color.y == 0 && color.z == 0)
				CountEachImageColData[id].w++;
		}
}
__global__ static void CountEachPixel(int4* CountEachImageColData, int4* CountEachImageData, int SmoothSizeRange, int maxid)
{
	int id = threadIdx.x * gridDim.x * gridDim.y +
		blockIdx.x * gridDim.y +
		blockIdx.y;

	if (id >= maxid)return;

	int halfSize = (SmoothSizeRange - 1) / 2;

	for (int jj = -halfSize; jj <= halfSize; jj++)
	{
		if (((blockIdx.x + jj) < 0) || ((blockIdx.x + jj) >= gridDim.x))
			continue;

		int idx = threadIdx.x * gridDim.x * gridDim.y +
			(blockIdx.x + jj) * gridDim.y +
			blockIdx.y;

		int4 count = CountEachImageColData[idx];

		CountEachImageData[id].x = count.x;
		CountEachImageData[id].y = count.y;
		CountEachImageData[id].z = count.z;
		CountEachImageData[id].w = count.w;
	}
}
__global__ static void SmoothEachPixel(int4* CountEachImageData, uchar3* SmoothData, int SmoothSizeRange, int maxid)
{
	int id = threadIdx.x * gridDim.x * gridDim.y +
		blockIdx.x * gridDim.y +
		blockIdx.y;

	if (id >= maxid)return;

	int halfSize = (SmoothSizeRange - 1) / 2;
	int CountArray[4] = { 0,0,0,0 };

	for (int ii = -halfSize; ii <= halfSize; ii++)
	{
		if (((threadIdx.x + ii) < 0) || ((threadIdx.x + ii) >= blockDim.x))
			continue;

		int idx = (threadIdx.x + ii) * gridDim.x * gridDim.y +
			blockIdx.x * gridDim.y +
			blockIdx.y;

		int4 count = CountEachImageData[idx];

		CountArray[3] += count.x;
		CountArray[2] += count.y;
		CountArray[1] += count.z;
		CountArray[0] += count.w;
	}

	int MaxIndex = 4;
	int maxValue[4] = { 0 };

	for (int i = 0; i < 4; i++)
		maxValue[i] = CountArray[i];

	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4 - i - 1; j++)
			if (CountArray[j] > CountArray[j + 1])
			{
				int tmp = CountArray[j];
				CountArray[j] = CountArray[j + 1];
				CountArray[j + 1] = tmp;
			}

	for (int i = 0; i < 4; i++)
		if (maxValue[i] == CountArray[3])
			MaxIndex = i;

	if (MaxIndex == 3) {
		SmoothData[id].x = 255;
		SmoothData[id].y = 0;
		SmoothData[id].z = 0;
	}
	else if (MaxIndex == 2) {
		SmoothData[id].x = 0;
		SmoothData[id].y = 255;
		SmoothData[id].z = 0;
	}
	else if (MaxIndex == 1) {
		SmoothData[id].x = 0;
		SmoothData[id].y = 0;
		SmoothData[id].z = 255;
	}
	else if (MaxIndex == 0) {
		SmoothData[id].x = 0;
		SmoothData[id].y = 0;
		SmoothData[id].z = 0;
	}
}

//////////////////////////////////////////////////////////////////////////
// CPU
//////////////////////////////////////////////////////////////////////////
void UtilityTools::SetImageData(vector<Mat> ImageData, int cMin, int  rMin, int cMax, int  rMax)
{
	int width = cMax - cMin + 1;	// cols
	int height = rMax - rMin + 1;	// rows

	// 總共幾個pixel
	int CutImageDataSize = ImageData.size() * height * width;	// 一張Image 有 height * width 個 pixel

	clock_t Utime = clock();
	// 上傳至GPU
	CPUImageData = new uchar3[CutImageDataSize];
	uchar3* GPUImageData;
	
	for (int i = 0; i < ImageData.size(); i++)
		for (int row = rMin; row < height; row++)
			for (int col = cMin; col < width; col++)
			{
				CPUImageData[i * height * width + row * width + col].x = ImageData[i].at<Vec3b>(row, col)[0];
				CPUImageData[i * height * width + row * width + col].y = ImageData[i].at<Vec3b>(row, col)[1];
				CPUImageData[i * height * width + row * width + col].z = ImageData[i].at<Vec3b>(row, col)[2];
			}

	Utime = clock() - Utime;
	cout << "上傳CPU花費時間 : " << ((float)Utime) / CLOCKS_PER_SEC << " sec" << endl;

	Utime = clock();

	cudaMalloc(&GPUImageData, sizeof(uchar3) * CutImageDataSize);
	cudaMemcpy(GPUImageData, CPUImageData, sizeof(uchar3) * CutImageDataSize, cudaMemcpyHostToDevice);
	CheckCudaError();

	// 計算每個 Col 的種類數量
	int4* CountEachImageColData;
	cudaMalloc(&CountEachImageColData, sizeof(int4) * CutImageDataSize);
	CountEachCol << < dim3(height, width, 1), ImageData.size() >> > (GPUImageData, CountEachImageColData, SmoothSizeRange, CutImageDataSize);
	CheckCudaError();

	cudaFree(GPUImageData);

	// 計算每個 Pixel 的種類數量
	int4* CountEachImageData;
	cudaMalloc(&CountEachImageData, sizeof(int4) * CutImageDataSize);
	CountEachPixel << < dim3(height, width, 1), ImageData.size() >> > (CountEachImageColData, CountEachImageData, SmoothSizeRange, CutImageDataSize);
	CheckCudaError(); 

	cudaFree(CountEachImageColData);

	// 計算每段 Pixel 的種類數量 及 Smooth
	uchar3* SmoothData;
	cudaMalloc(&SmoothData, sizeof(uchar3) * CutImageDataSize);
	SmoothEachPixel << < dim3(height, width, 1), ImageData.size() >> > (CountEachImageData, SmoothData, SmoothSizeRange, CutImageDataSize);
	CheckCudaError();

	cudaFree(CountEachImageData);

	// 刪除之前的資料
	SaveDelete(CPUImageData);
	CPUImageData = new uchar3[CutImageDataSize];
	cudaMemcpy(CPUImageData, SmoothData, sizeof(uchar3) * CutImageDataSize, cudaMemcpyDeviceToHost);

	cudaFree(SmoothData);

	Utime = clock() - Utime;
	cout << "GPU花費時間 : " << ((float)Utime) / CLOCKS_PER_SEC << " sec" << endl;

	size = ImageData.size();
	rows = height;
	cols = width;
}

// 拿出圖片
vector<Mat> UtilityTools::TransfromMatArray()
{
	// 轉換到 Mat
	vector<Mat> ImgArray;
	for (int i = 0; i < size; i++)
	{
		// 根據 Offset 拿圖片
		Mat img(rows, cols, CV_8UC3, CPUImageData + i * rows * cols);

		// 丟進堆疊
		ImgArray.push_back(img);
	}
	return ImgArray;
}

//////////////////////////////////////////////////////////////////////////
// Helper Function
//////////////////////////////////////////////////////////////////////////
void UtilityTools::CheckCudaError()
{
	cudaError GPU_Error = cudaGetLastError();
	if (GPU_Error != cudaSuccess)
	{
		cout << cudaGetErrorString(GPU_Error) << endl;
		assert(false);
		exit(-1);
	}
}
void UtilityTools::SaveDelete(void* pointer)
{
	if (pointer != NULL)
		delete[] pointer;
}