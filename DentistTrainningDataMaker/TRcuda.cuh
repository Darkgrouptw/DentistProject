#ifndef TRCUDA_H
#define TRCUDA_H

#include <vector>
#include <iostream>
#include <time.h>

#include <cuda_runtime.h>
#include <cufft.h>

//using namespace std;

class TRcuda
{
public:
	TRcuda(void);
	~TRcuda(void);

	bool InitCUDA();
	//void CharToShortRawToPC(char* rawData, int data_Size, int size_Y, int size_Z, int sample);
	void RawToPointCloud(char* rawData, int data_Size, int size_Y, int size_Z, int ch = 2);
	void RawToSingle(char* rawData, int data_Size, int size_Y, int size_Z, int ch = 2);
	void Test();

	std::vector<int> tPointCloud;

	int avergeBlock; // [ -avergeBlock ~ 0 ~ avergeBlock ]
	float peakGap;
	float energyGap;
	int depthFRange;
	int depthBRange;
	int boardNRange;
	int boardSpread;
	int cut_X;
	int cut_Z;
	int shift_X;
	int sample_X;
	int sample_Y;
	int split;
	float* CalibrationMap;
	float radiusRange;

	float* VolumeData;
	float* SingleData;
	float* VolumeDataAvg;
	float* RawDataScanP;
	int* PointType;

	int VolumeSize_X;
	int VolumeSize_Y;
	int VolumeSize_Z;
};

#endif // TRCUDA_H