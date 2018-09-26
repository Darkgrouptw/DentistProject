#include "TRcuda.cuh"
#include "device_launch_parameters.h"
#include "Math.h"

__global__ static void gpuRawDataToFloat(short* in, float* out, int dataSize)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;

	for (int i = Id; i < dataSize; i += threadNum) 
	{
		out[i] = in[i] - 32768;
	}
}

__global__ static void gpuRawDataToFloat(char* in, float* out, int dataSize)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;

	int tmp;

	for (int i = Id; i < dataSize; i += threadNum)
	{
		//out[i] = in[i * 2] + in[(i * 2) + 1] * 256 - 32768;
		out[i] = in[i * 2] + in[(i * 2) + 1] * 256;
		//*
		if (out[i] < 0)
			out[i] = 65536 + out[i];
		//*/
		out[i] = out[i] - 32768;
	}
}

__global__ static void gpu2RawDataToFloat(char* in, float* out, int dataSize, int size_Y, int size_Z)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;

	int tmp, idx0, idx1, sline, spoint, scanline = size_Z * size_Y * 2;

	for (int i = Id; i < dataSize; i += threadNum)
	{
		sline = i / scanline;
		spoint = i % scanline;
		idx0 = (sline * scanline * 2 + spoint) * 2;
		idx1 = (sline * scanline * 2 + scanline + spoint) * 2;

		//out[i] = abs(in[idx0] + in[idx0 + 1] * 256) + abs(in[idx1] + in[idx1 + 1] * 256);
		//out[i] = out[i] / 2 - 32768;
		
		
		out[i] = (in[idx0] + in[idx0 + 1] * 256);
		tmp = (in[idx1] + in[idx1 + 1] * 256);
		//*
		if (out[i] < 0)
			out[i] = out[i] + 65536;
		if (tmp < 0)
			tmp = tmp + 65536;
		//*/
		out[i] = (out[i] + tmp)/2 - 32768;
	}
}

__global__ static void gpuCutUselessDataX(float* in, float* out, int dataSize, int size_X, int size_Y, int size_Z, int cutX)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;

	if (cutX > 0)
	{
		int idx;
		for (int i = Id; i < dataSize; i += threadNum)
		{
			idx = cutX * (size_Y * size_Z);
			out[i] = in[i + idx];
		}
	}
	else {
		cutX = -cutX;
		for (int i = Id; i < dataSize; i += threadNum)
		{
			//idx = cutX * (size_Y * size_Z);
			out[i] = in[i];
		}
	}
}

__global__ static void gpuCutUselessDataZ(float* in, float* out, int dataSize, int size_X, int size_Y, int size_Z, int cutZ)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;

	int idx, nowXY;
	for (int i = Id; i < dataSize; i += threadNum)
	{
		nowXY = i / size_Z;
		idx = i + nowXY * cutZ;
		out[i] = in[idx];
	}
}

__global__ static void gpuDataReverse(float* in, float* out, int dataSize, int size_Y, int size_Z)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;
	int tmp, nowY;

	for (int i = Id; i < dataSize; i += threadNum)
	{
		tmp = i / (size_Y * size_Z);
		if (tmp % 2 == 0)
		{
			out[i] = in[i];
		}
		else // need reverse
		{
			nowY = i / size_Z;
			nowY = nowY % size_Y;
			nowY = size_Y - nowY - 1;
			out[(tmp * size_Y + nowY) * size_Z + i % size_Z] = in[i];
		}
	}
}

__global__ static void gpuSampleXY(float* in, float* out, int dataSize, int size_Y, int size_Z, int sample_X, int sample_Y)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;
	//*
	int idx, idy, idz;
	for (int i = Id; i < dataSize / (sample_X * sample_Y); i += threadNum)
	{
		idx = (i / size_Z) / (size_Y / sample_Y);
		idy = (i / size_Z) % (size_Y / sample_Y);
		idz = i % size_Z;
		out[i] = in[((idx * (sample_X) * size_Y) + idy * (sample_Y)) * size_Z + idz];
	}
	//*/
}

__global__ static void gpuDataToComplex(float* in, cufftComplex* out, int dataSize, int start)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;

	for (int i = Id; i < dataSize; i += threadNum)
	{
		out[i].x = in[start+i];
		out[i].y = 0;
	}
}

__global__ static void gpuComplexToData(cufftComplex* in, float* out, int dataSize, int size_Z, int start)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;
	float tmp;
	int newId;

	for (int i = Id; i < dataSize; i += threadNum)
	{
		newId = (i / size_Z) * size_Z + i; // cut mirror data
		tmp = sqrt(in[newId].x * in[newId].x + in[newId].y * in[newId].y);
		out[start+i] = log10f(tmp);
	}
}

__global__ static void gpuShift(float* in, float* out, int dataSize, int size_X, int size_Y, int size_Z, int shiftX)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;
	int nowX, nowYZ;

	for (int i = Id; i < dataSize; i += threadNum)
	{
		nowX = i / (size_Z * size_Y);
		nowYZ = i % (size_Z * size_Y);
		out[i] = in[((nowX + shiftX) % size_X)*size_Y*size_Z + nowYZ];
	}
}

__global__ static void gpuZaxisAverge(float* in, float* out, int dataSize, int size_Z, int block)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;
	int tmp;
	float sum, count;

	for (int i = Id; i < dataSize; i += threadNum)
	{
		tmp = i % size_Z;
		sum = 0;
		count = 0;
		for (int z = -block; z < block; z++)
		{
			if (tmp + z >= 0 && tmp + z < size_Z)
			{
				sum += in[i + z];
				count++;
			}
		}
		out[i] = sum / count;
	}
}

__global__ static void gpuSumThresh(float* in, int dataSize, int size_Z, int dFRange, int dBRange)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;
	float sum, max;
	for (int i = Id * size_Z; i < dataSize; i += threadNum * size_Z)
	{
		sum = 0;
		max = 0;
		for (int d = i + dFRange; d + dBRange < (i + size_Z); d++)
		{
			sum += in[d];
			if (in[d] > max)
				max = in[d];
		}
		in[i] = sum;
		in[i + size_Z - 1] = max;
	}
}

__global__ static void gpuPeakDetect(float* in, int* out, int dataSize, int size_Z, float peakGap, float energyGap, int dFRange, int dBRange)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;

	int lmaxid, lminid, count, nstate, nowX, nowY;
	for (int i = Id * size_Z; i < dataSize; i += threadNum * size_Z)
	{
		//* SumThresh
		if (in[i] <= in[0] * 1.001)
		{
			out[i] = -1;
			continue;
		}//*/
		lmaxid = i;
		lminid = i;
		count = 0;
		nstate = 0;// 1:up -1:down

		for (int d = i + dFRange; d + dBRange < (i + size_Z); d++)
		{
			if (in[d - 1] < in[d]) // up
			{
				if (nstate == -1) // down peak
				{
					//if (in[lmaxid] - in[d - 1] > 0.1)
					{
						out[d - 1] = 5;
						lminid = d - 1;
					}
				}
				nstate = 1;
			}
			else if (in[d - 1] > in[d]) // down
			{
				if (nstate == 1) // up peak
				{
					out[d - 1] = 2;
					lmaxid = d - 1;
					if (in[d - 1] - in[lminid] > peakGap)
					{
						if (in[lmaxid] > energyGap)// Gap check
						{
							out[d - 1] = 3;
							if (in[lmaxid + 2] > energyGap && in[lmaxid + 4] > energyGap)// Continuous check
							{
								out[d - 1] = 1;
								count++;
								if (count == 1)
									out[i + 1] = d - i - 1;
							}
						}
					}
				}
				nstate = -1;
			}
		}
		// z axis 0 = peak count
		//        1 = first peak idx (0 ~ size_Z-1) or board peak idx or temp compare idx
		//        2 = init board : -1 = t, 0 = f
		//        3 = found board : -1 = no board, 0 = need find, 1 = found
		//        4 = neighbor min dist
		//        5 = neighbor board check state : -1 = checked, 0 = finding neighbor, 1 = finding nearist board
		//        6 = neighbor board idx up
		//        7 = neighbor board idx down
		//        8 = neighbor board idx left
		//        9 = neighbor board idx right
		// else 2 = up peak, 3 = peak > gap, 1 = peak Continuous
		out[i] = count;
	}
}

__global__ static void gpuFindMinPeak(int* in, int* out, int size_X, int size_Y, int size_Z)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;
	int minPeak, idx;
	for (int i = Id; i < size_X; i += threadNum)
	{
		minPeak = size_Z;
		for (int d = 0; d < size_Y; d++)
		{
			idx = ((i * size_Y) + d) * size_Z;
			if (in[idx] > 0)
			{
				if (in[idx] < minPeak)
					minPeak = in[idx];
			}
		}
		out[i] = minPeak;
	}
}

__global__ static void gpuFindMinPeakOne(int* in, int size_X)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;
	int minPeak;
	if (Id == 0)
	{
		minPeak = size_X;
		for (int i = Id; i < size_X; i++)
		{
			if (in[i] > 0)
			{
				if (in[i] < minPeak)
					minPeak = in[i];
			}
		}
		in[0] = minPeak;
	}
}

__global__ static void gpuSetMinPeak(int* in, int dataSize, int size_X, int size_Y, int size_Z, int minPeak, int nRange)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;
	int pCount;

	for (int i = Id * size_Z; i < dataSize; i += threadNum * size_Z)
	{
		/*
		if (in[i] == minPeak)
		{
			in[i + 2] = -1;
			in[i + 3] = 1;
			in[i + 5] = -1;
		}
		if (in[i] == 0)
		{
			in[i + 3] = -1;
			in[i + 5] = -1;
		}
		//*/
		//*
		if (in[i] == minPeak)
		{
			pCount = 0;
			if (i%size_Y >= 1 && in[i - size_Z] == minPeak)
			{
				if(abs(in[i + 1] - in[i - size_Z + 1]) < nRange)
					pCount++;
			}if (i%size_Y < size_Y - 1 && in[i + size_Z] == minPeak)
			{
				if (abs(in[i + 1] - in[i + size_Z + 1]) < nRange)
					pCount++;
			}
			if (i / size_Y / size_Z >= 1 && in[i - size_Z*size_Y] == minPeak)
			{
				if (abs(in[i + 1] - in[i - size_Z*size_Y + 1]) < nRange)
					pCount++;
			}if (i / size_Y / size_Z < size_X - 1 && in[i + size_Z*size_Y + 3] == minPeak)
			{
				if (abs(in[i + 1] - in[i + size_Z*size_Y + 1]) < nRange)
					pCount++;
			}
			if (pCount++ >= 1)// neighbor minpeak count
			{
				in[i + 2] = -1;
				in[i + 3] = 1;
				in[i + 5] = -1;
			}
		}
		else if (in[i] == 0)
		{
			in[i + 3] = -1;
			in[i + 5] = -1;
		}
		//*/
	}
}

__global__ static void gpuBoardDetect(int* in, int dataSize, int size_X, int size_Y, int size_Z, int dFRange, int dBRange, int nRange)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;

	int compareId, min, minId = 0, tmpMin;
	for (int i = Id * size_Z; i < dataSize; i += threadNum * size_Z)
	{
		// z axis 3 = found board : -1 = no board, 0 = need find, 1 = found
		//        5 = neighbor board check state : -1 = find end, 0 = finding neighbor, 1 = finding nearist board

		if (in[i + 5] == 1)// find nearest peak
		{
			min = size_Z;
			if (in[i + 6] > 0)
			{
				compareId = in[i + 6];
				for (int d = i + dFRange; d + dBRange < (i + size_Z); d++)
				{
					if (in[d] == 1)
					{
						tmpMin = abs(d - i - compareId);
						if (tmpMin < min)
						{
							min = tmpMin;
							minId = d - i;
						}
					}
				}
				in[i + 6] = -in[i + 6];
			}
			if (in[i + 7] > 0)
			{
				compareId = in[i + 7];
				for (int d = i + dFRange; d + dBRange < (i + size_Z); d++)
				{
					if (in[d] == 1)
					{
						tmpMin = abs(d - i - compareId);
						if (tmpMin < min)
						{
							min = tmpMin;
							minId = d - i;
						}
					}
				}
				in[i + 7] = -in[i + 7];
			}
			if (in[i + 8] > 0)
			{
				compareId = in[i + 8];
				for (int d = i + dFRange; d + dBRange < (i + size_Z); d++)
				{
					if (in[d] == 1)
					{
						tmpMin = abs(d - i - compareId);
						if (tmpMin < min)
						{
							min = tmpMin;
							minId = d - i;
						}
					}
				}
				in[i + 8] = -in[i + 8];
			}
			if (in[i + 9] > 0)
			{
				compareId = in[i + 9];
				for (int d = i + dFRange; d + dBRange < (i + size_Z); d++)
				{
					if (in[d] == 1)
					{
						tmpMin = abs(d - i - compareId);
						if (tmpMin < min)
						{
							min = tmpMin;
							minId = d - i;
						}
					}
				}
				in[i + 9] = -in[i + 9];
			}
			in[i + 5] = 0;
			if (min < nRange)
			{
				if (in[i + 4] == 0 || min + 1 < in[i + 4])
				{
					in[i + 4] = min + 1;
					in[i + 1] = minId;
					in[i + 3] = 1;
				}
			}
			else if(in[i + 6] < 0 && in[i + 7] < 0 && in[i + 8] < 0 && in[i + 9] < 0)
			{
				in[i + 5] = -1;
			}
		}
		else if (in[i + 5] == 0)// find neighbor board
		{
			if (i%size_Y >= 1 && in[i + 6] == 0)
			{
				if (in[i - size_Z + 3] == 1)
				{
					in[i + 6] = in[i - size_Z + 1];
					in[i + 5] = 1;
				}
			}if (i%size_Y < size_Y - 1 && in[i + 7] == 0)
			{
				if (in[i + size_Z + 3] == 1)
				{
					in[i + 7] = in[i + size_Z + 1];
					in[i + 5] = 1;
				}
			}
			if (i / size_Y / size_Z >= 1 && in[i + 8] == 0)
			{
				if (in[i - size_Z*size_Y + 3] == 1)
				{
					in[i + 8] = in[i - size_Z*size_Y + 1];
					in[i + 5] = 1;
				}
			}if (i / size_Y / size_Z < size_X - 1 && in[i + 9] == 0)
			{
				if (in[i + size_Z*size_Y + 3] == 1)
				{
					in[i + 9] = in[i + size_Z*size_Y + 1];
					in[i + 5] = 1;
				}
			}
			if (in[i + 6] < 0 && in[i + 7] < 0 && in[i + 8] < 0 && in[i + 9] < 0)
			{
				in[i + 5] = -1;
			}
		}
	}
}

__global__ static void gpuMapping(int* in, int* out, int size_X, int size_Y, int size_Z)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;
	//*
	int minPeak, idx;
	if (Id == 0)
	{
		for (int i = Id; i < size_X; i ++)
		{
			minPeak = size_Z;
			for (int d = 0; d < size_Y; d++)
			{
				idx = ((i * size_Y) + d) * size_Z;
				if (in[idx] > 0)
				{
					out[idx] = in[idx];
				}
			}
			//out[i] = minPeak;
		}
	}
	//*/
}

__global__ static void gpuComputePXScale(double* out, int size_Z)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;
	long x1 = 1;
	long zl = 2.2512322 + 0.3954103 * x1 - 0.000011 * pow(x1 - 997.111, 2) - 0.000000010542 * pow(x1 - 997.111, 3) - 0.000000000005322 * pow(x1 - 997.111, 4) - 0.000000000000001312 * pow(x1 - 997.111, 5) + 2.984E-18 * pow(x1 - 997.111, 6);
	//double x1 = 1;
	//double z = 2.2512322 + 0.3954103 * x1 - 0.000011 * pow(x1 - 997.111, 2) - 0.000000010542 * pow(x1 - 997.111, 3) - 0.000000000005322 * pow(x1 - 997.111, 4) - 0.000000000000001312 * pow(x1 - 997.111, 5) + 2.984E-18 * pow(x1 - 997.111, 6);

	for (int i = Id; i < size_Z; i += threadNum)
	{
		x1 = i + 1;
		out[i] = 2.2512322 + 0.3954103 * x1 - 0.000011 * pow(x1 - 997.111, 2) - 0.000000010542 * pow(x1 - 997.111, 3) - 0.000000000005322 * pow(x1 - 997.111, 4) - 0.000000000000001312 * pow(x1 - 997.111, 5) + 2.984E-18 * pow(x1 - 997.111, 6);
		out[i] = out[i] - zl;
	}
}

__global__ static void gpuComputeStepVal(double* in, int* out, int size_Z)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;

	double step = 2.5, v = 1 / step, x1 = (double)size_Z, z = 2.2512322 + 0.3954103 * x1 - 0.000011 * pow(x1 - 997.111, 2) - 0.000000010542 * pow(x1 - 997.111, 3) - 0.000000000005322 * pow(x1 - 997.111, 4) - 0.000000000000001312 * pow(x1 - 997.111, 5) + 2.984E-18 * pow(x1 - 997.111, 6);
	int x = 1, y = 1;
	for (int i = Id; i < size_Z - 1; i += threadNum)
	{
		v = 1 / step * (double)(i + 1);
		for (y = 0; y < size_Z - 1; y++)
		{
			if (v < in[y])
			{
				break;
			}
		}
		if (v > z || y > size_Z - 1)
		{
			break;
		}
		out[i] = y;
	}
}

__global__ static void gpuStepValCheck(int* in, int size_Z)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;

	if (Id == 0)
	{
		for (int i = Id + 1; i < size_Z; i++)
		{
			if (in[i] == in[i - 1] || in[i - 1] == 0)
			{
				if (in[i] == 0)
				{
					break;
				}
				in[i] = 0;
			}
		}
	}
}

__global__ static void gpuFrequencyAdjust(int* in, float* out, int size_Z, int dataSize)
{
	// gridDim.x blockIdx.x blockDim.x threadIdx.x
	int Id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNum = gridDim.x * blockDim.x;

	int XY = dataSize / size_Z;

	for (int i = Id; i < XY; i += threadNum)
	{
		for (int j = 0; j < size_Z; j++)
		{
			if (in[j] == 0)
			{
				out[i * size_Z + j] = 0;
			}
			else
			{
				out[i * size_Z + j] = out[i * size_Z + in[j]];
			}
		}
	}
}
// ^^^^^^^^^^^^^^^^^^^^ GPU ^^^^^^^^^^^^^^^^^^^^
// vvvvvvvvvvvvvvvvvvvv CPU vvvvvvvvvvvvvvvvvvvv
TRcuda::TRcuda(void)
{
	// variable init
	avergeBlock = 5;//3
	peakGap = 0.1f;//0.1//0.3
	energyGap = 4.25f; //6.0//4.1 4.4f
	depthFRange = 30;// abandon front > 10
	depthBRange = 400;//270 abandon back
	boardNRange = 10;// near range
	boardSpread = 100;
	cut_X = 0; // >0 = cut front, <0 = cut back (62, -82) 296 -150
	cut_Z = 0;// 352 400
	shift_X = 125;// 200 125
	sample_X = 1;//2
	sample_Y = 1;//2
	split = 1;//5
	radiusRange = 125;

	CalibrationMap = NULL;
	VolumeData = NULL;
	SingleData = NULL;
	VolumeDataAvg = NULL;
	RawDataScanP = NULL;
	PointType = NULL;

	// init function
	InitCUDA();
}


TRcuda::~TRcuda(void)
{
	if (VolumeData != NULL)
		free(VolumeData);
	if (VolumeDataAvg != NULL)
		free(VolumeDataAvg);
	if (PointType != NULL)
		free(PointType);
	if (SingleData != NULL)
		free(SingleData);
}

bool TRcuda::InitCUDA()
{
	std::cout << "InitCUDA \n";
	int count;

	cudaGetDeviceCount(&count);
	if (count == 0) {
		std::cout << "There is no device\n";
		return false;
	}

	int i;
	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}

	if (i == count) {
		std::cout << "There is no device supporting CUDA 1.x.\n";
		return false;
	}

	cudaSetDevice(i);
	std::cout << "Use device : " << i + 1 << " / " << count << "\n";

	return true;
}
/*
void TRcuda::CharToShortRawToPC(char * rawData, int data_Size, int size_Y, int size_Z, int sample)
{

}
//*/
void TRcuda::RawToSingle(char* rawData, int data_Size, int size_Y, int size_Z, int ch) {
	clock_t t1, t2, t3, t4;
	cudaError gpuError;
	//int blockDimX = 512, gridDimX = 4, debugNum = 20, debugPos = 249 * 250 * 2048 + 1014;
	//
	//t1 = clock();
	//t3 = t1;
	//char* gpuData;
	//// load to GPU
	//gpuError = cudaMalloc((void**)&gpuData, sizeof(char) *data_Size);
	//gpuError = cudaMemcpy(gpuData, rawData, sizeof(char) *data_Size, cudaMemcpyHostToDevice);
	////free(rawData);
	//
	//int floatDataSize = data_Size;
	//int size_X = floatDataSize / size_Z / size_Y / 2;
	//t2 = clock();
	//std::cout << "MemcpyHostToDevice done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";
	//
	//// Raw data to float
	//t1 = clock();
	//floatDataSize = floatDataSize / 2;// 2 char to 1 float
	//float* gpuFloatData;
	//gpuError = cudaMalloc((void**)&gpuFloatData, sizeof(float) * floatDataSize);
	//
	//if (ch == 2)
	//{
	//	floatDataSize = floatDataSize / 2;// 2 channel to 1 channel
	//	size_X = size_X / 2;// 2 channel to 1 channel
	//	gpu2RawDataToFloat << <gridDimX, blockDimX >> > (gpuData, gpuFloatData, floatDataSize, size_Y, size_Z);
	//}
	//else
	//	gpuRawDataToFloat << <gridDimX, blockDimX >> > (gpuData, gpuFloatData, floatDataSize);//1 channel
	//
	//gpuError = cudaDeviceSynchronize();
	//cudaFree(gpuData);
	//
	//t2 = clock();
	//std::cout << "floatDataSize : " << floatDataSize << "\n";
	//std::cout << "RawDataToFloat done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";
	//
	////if (cut_X != 0 || cut_Z != 0)
	////{
	////	t1 = clock();
	////	float* gpuCutDataZ;
	////	/*floatDataSize = (floatDataSize / size_X) * (size_X - abs(cut_X));
	////	size_X = size_X - abs(cut_X);
	////	gpuError = cudaMalloc((void**)&gpuCutDataX, sizeof(float) * floatDataSize);
	////
	////	gpuCutUselessDataX << <gridDimX, blockDimX >> > (gpuFloatData, gpuCutDataX, floatDataSize, size_X, size_Y, size_Z, cut_X);
	////	gpuError = cudaDeviceSynchronize();
	////
	////	cudaFree(gpuFloatData);*/
	////
	////	// Cut Z
	////	//float* gpuCutDataZ;
	////	floatDataSize = (floatDataSize / size_Z) * (size_Z - abs(cut_Z));
	////	size_Z = size_Z - abs(cut_Z);
	////	gpuError = cudaMalloc((void**)&gpuFloatData, sizeof(float) * floatDataSize);
	////
	////	gpuCutUselessDataZ << <gridDimX, blockDimX >> > (gpuFloatData, gpuCutDataZ, floatDataSize, size_X, size_Y, size_Z, cut_Z);
	////	gpuError = cudaDeviceSynchronize();
	////
	////	cudaFree(gpuFloatData);
	////
	////	t2 = clock();
	////	std::cout << "floatDataSize : " << floatDataSize << "\n";
	////	std::cout << "CutUselessData done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";
	////
	////	/* debug output : Cut
	////	cout<<"gpuError : "<<cudaGetErrorString(gpuError)<<"\n";
	////	float* debugFloatData = (float*) malloc(sizeof(float) * floatDataSize);
	////	gpuError = cudaMemcpy(debugFloatData, gpuFloatData, sizeof(int) * floatDataSize, cudaMemcpyDeviceToHost);
	////	for(int i = 0; i < debugNum; i++)
	////	{
	////	cout<<"idx : "<< i + debugPos <<" floatData : "<<debugFloatData[i + debugPos]<<"\n";
	////	}
	////	free(debugFloatData);
	////	//*/
	////
	////
	////}
	//// Reverse data
	////*
	//t1 = clock();
	//float* gpuReversedData;
	//gpuError = cudaMalloc((void**)&gpuReversedData, sizeof(float) *floatDataSize);
	//
	//gpuDataReverse << <gridDimX, blockDimX >> >(gpuFloatData, gpuReversedData, floatDataSize, size_Y, size_Z);
	//gpuError = cudaDeviceSynchronize();
	//cudaFree(gpuFloatData);
	//
	//t2 = clock();
	////cout << "Reverse size : " << floatDataSize << "\n";
	//std::cout << "Reverse done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";
	//
	//
	////lambda to k
	////*
	//t1 = clock();
	//double* gpuPXscale;
	//int* gpuStepVal;
	//gpuError = cudaMalloc((void**)&gpuPXscale, sizeof(double) * size_Z);
	//gpuError = cudaMalloc((void**)&gpuStepVal, sizeof(int) * size_Z);
	//
	//gpuComputePXScale << <gridDimX, blockDimX >> > (gpuPXscale, size_Z);
	//gpuError = cudaDeviceSynchronize();
	//
	//gpuComputeStepVal << <gridDimX, blockDimX >> > (gpuPXscale, gpuStepVal, size_Z);
	//gpuError = cudaDeviceSynchronize();
	//
	//gpuStepValCheck << <gridDimX, blockDimX >> > (gpuStepVal, size_Z);
	//gpuError = cudaDeviceSynchronize();
	//
	////*/
	//gpuFrequencyAdjust << <gridDimX, blockDimX >> > (gpuStepVal, gpuReversedData, size_Z, floatDataSize);//gpuCutDataZ
	//gpuError = cudaDeviceSynchronize();
	////cout << "gpuError : " << cudaGetErrorString(gpuError) << "\n";
	//cudaFree(gpuPXscale);
	//cudaFree(gpuStepVal);
	//t2 = clock();
	//std::cout << "lambda to k done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";
	////*/
	//
	//// Sample X Y
	////*
	//bool hadSample = sample_X != 1 || sample_Y != 1;
	//float* gpuSampleData;
	//if (hadSample)
	//{
	//	t1 = clock();
	//
	//	gpuError = cudaMalloc((void**)&gpuSampleData, sizeof(float) * floatDataSize);
	//
	//	gpuSampleXY << <gridDimX, blockDimX >> >(gpuReversedData, gpuSampleData, floatDataSize, size_Y, size_Z, sample_X, sample_Y);
	//	gpuError = cudaDeviceSynchronize();
	//
	//	floatDataSize = floatDataSize / (sample_X * sample_Y);
	//	size_X = size_X / sample_X;
	//	size_Y = size_Y / sample_Y;
	//
	//	cudaFree(gpuReversedData);
	//
	//	t2 = clock();
	//	//cout << "gpuError : " << cudaGetErrorString(gpuError) << "\n";
	//	std::cout << "floatDataSize : " << floatDataSize << "\n";
	//	std::cout << "SampleXY done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";
	//}//*/
	//
	// // CUFFT
	//t1 = clock();
	//cufftHandle plan;
	//cufftComplex *gpuComplexData;
	//int NX = size_Z, BATCH = floatDataSize / size_Z;
	//
	//floatDataSize = floatDataSize / 2;
	//size_Z = size_Z / 2;
	//
	////split for reduce max gpu memory use
	//BATCH = BATCH / split;
	//
	//gpuError = cudaMalloc((void**)&gpuFloatData, sizeof(float) * floatDataSize);
	//for (int i = 0; i < split; i++)
	//{
	//	// set cufftplan
	//	cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH);
	//	gpuError = cudaMalloc((void**)&gpuComplexData, sizeof(cufftComplex) * NX * BATCH);
	//
	//	// data in  gpuCutDataZ
	//	if (hadSample)
	//		gpuDataToComplex << <gridDimX, blockDimX >> > (gpuSampleData, gpuComplexData, NX * BATCH, NX * BATCH * i);
	//	else
	//		gpuDataToComplex << <gridDimX, blockDimX >> > (gpuReversedData, gpuComplexData, NX * BATCH, NX * BATCH * i);
	//	gpuError = cudaDeviceSynchronize();
	//
	//	// excute cufft
	//	cufftExecC2C(plan, gpuComplexData, gpuComplexData, CUFFT_FORWARD);
	//	gpuError = cudaDeviceSynchronize();
	//
	//	// data out and cut mirror data
	//	gpuComplexToData << <gridDimX, blockDimX >> > (gpuComplexData, gpuFloatData, NX * BATCH / 2, size_Z, NX * BATCH / 2 * i);
	//	gpuError = cudaDeviceSynchronize();
	//
	//	cufftDestroy(plan);
	//	cudaFree(gpuComplexData);
	//}
	//if (hadSample)
	//	cudaFree(gpuSampleData);
	//else
	//	cudaFree(gpuReversedData);
	//t2 = clock();
	//std::cout << "cuff nx : " << NX << " Batch : " << BATCH << " split : " << split << "\n";
	//std::cout << "CUFFT done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";
	//
	///* debug output : CUFFT
	//cout<<"gpuError : "<<cudaGetErrorString(gpuError)<<"\n";
	//float* debugCUFFTData = (float*) malloc(sizeof(float) * floatDataSize);
	//gpuError = cudaMemcpy(debugCUFFTData, gpuCutDataZ, sizeof(float) * floatDataSize, cudaMemcpyDeviceToHost);
	//for(int i = 0; i < debugNum; i++)
	//{
	//cout<<"idx : "<< i + debugPos <<" floatData : "<<debugCUFFTData[i + debugPos]<<"\n";
	//}
	//free(debugCUFFTData);
	////*/
	//// data out for opengl draw
	//if (SingleData != NULL)
	//	free(SingleData);
	//SingleData = (float*)malloc(sizeof(float) * floatDataSize);
	//gpuError = cudaMemcpy(SingleData, gpuFloatData, sizeof(float) * floatDataSize, cudaMemcpyDeviceToHost);

	
}

void TRcuda::RawToPointCloud(char* rawData, int data_Size, int size_Y, int size_Z, int ch)
{
	clock_t t1, t2, t3, t4;
	cudaError gpuError;
	int blockDimX = 512, gridDimX = 4, debugNum = 20, debugPos = 249 * 250 * 2048 + 1014;

	t1 = clock();
	t3 = t1;
	char* gpuData;
	// load to GPU
	gpuError = cudaMalloc((void**)&gpuData, sizeof(char) *data_Size);
	gpuError = cudaMemcpy(gpuData, rawData, sizeof(char) *data_Size, cudaMemcpyHostToDevice);
	//free(rawData);

	int floatDataSize = data_Size;
	int size_X = floatDataSize / size_Z / size_Y / 2;
	t2 = clock();
	std::cout << "MemcpyHostToDevice done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";

	// Raw data to float
	t1 = clock();
	floatDataSize = floatDataSize / 2;// 2 char to 1 float
	float* gpuFloatData;
	gpuError = cudaMalloc((void**)&gpuFloatData, sizeof(float) * floatDataSize);

	if (ch == 2)
	{
		floatDataSize = floatDataSize / 2;// 2 channel to 1 channel
		size_X = size_X / 2;// 2 channel to 1 channel
		gpu2RawDataToFloat << <gridDimX, blockDimX >> > (gpuData, gpuFloatData, floatDataSize, size_Y, size_Z);
	}
	else
		gpuRawDataToFloat << <gridDimX, blockDimX >> > (gpuData, gpuFloatData, floatDataSize);//1 channel

	gpuError = cudaDeviceSynchronize();
	cudaFree(gpuData);

	t2 = clock();
	std::cout << "floatDataSize : " << floatDataSize << "\n";
	std::cout << "RawDataToFloat done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";

	/* debug output : Raw data to float
	cout<<"gpuError : "<<cudaGetErrorString(gpuError)<<"\n";
	float* debugFloatData = (float*) malloc(sizeof(float) * floatDataSize);
	gpuError = cudaMemcpy(debugFloatData, gpuFloatData, sizeof(int) * floatDataSize, cudaMemcpyDeviceToHost);
	for(int i = 0; i < debugNum; i++)
	{
	cout<<"idx : "<< i + debugPos <<" floatData : "<<debugFloatData[i + debugPos]<<"\n";
	}
	free(debugFloatData);
	//*/

	// Cut useless data
	//Cut X
	if (cut_X != 0 || cut_Z != 0)
	{
		t1 = clock();
		float* gpuCutDataX;
		floatDataSize = (floatDataSize / size_X) * (size_X - abs(cut_X));
		size_X = size_X - abs(cut_X);
		gpuError = cudaMalloc((void**)&gpuCutDataX, sizeof(float) * floatDataSize);

		gpuCutUselessDataX << <gridDimX, blockDimX >> > (gpuFloatData, gpuCutDataX, floatDataSize, size_X, size_Y, size_Z, cut_X);
		gpuError = cudaDeviceSynchronize();

		cudaFree(gpuFloatData);

		// Cut Z
		//float* gpuCutDataZ;
		floatDataSize = (floatDataSize / size_Z) * (size_Z - abs(cut_Z));
		size_Z = size_Z - abs(cut_Z);
		gpuError = cudaMalloc((void**)&gpuFloatData, sizeof(float) * floatDataSize);

		gpuCutUselessDataZ << <gridDimX, blockDimX >> > (gpuCutDataX, gpuFloatData, floatDataSize, size_X, size_Y, size_Z, cut_Z);
		gpuError = cudaDeviceSynchronize();

		cudaFree(gpuCutDataX);

		t2 = clock();
		std::cout << "floatDataSize : " << floatDataSize << "\n";
		std::cout << "CutUselessData done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";

		/* debug output : Cut
		cout<<"gpuError : "<<cudaGetErrorString(gpuError)<<"\n";
		float* debugFloatData = (float*) malloc(sizeof(float) * floatDataSize);
		gpuError = cudaMemcpy(debugFloatData, gpuFloatData, sizeof(int) * floatDataSize, cudaMemcpyDeviceToHost);
		for(int i = 0; i < debugNum; i++)
		{
		cout<<"idx : "<< i + debugPos <<" floatData : "<<debugFloatData[i + debugPos]<<"\n";
		}
		free(debugFloatData);
		//*/
	}

	// Reverse data
	//*
	t1 = clock();
	float* gpuReversedData;
	gpuError = cudaMalloc((void**)&gpuReversedData, sizeof(float) *floatDataSize);

	gpuDataReverse << <gridDimX, blockDimX >> >(gpuFloatData, gpuReversedData, floatDataSize, size_Y, size_Z);
	gpuError = cudaDeviceSynchronize();
	cudaFree(gpuFloatData);

	t2 = clock();
	//cout << "Reverse size : " << floatDataSize << "\n";
	std::cout << "Reverse done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";
	//*/
	/* debug output : Reverse data
	cout<<"gpuError : "<<cudaGetErrorString(gpuError)<<"\n";
	float* debugFloatReverseData = (float*) malloc(sizeof(float) * floatDataSize);
	gpuError = cudaMemcpy(debugFloatReverseData, gpuReversedData, sizeof(float) * floatDataSize, cudaMemcpyDeviceToHost);
	for(int i = 0; i < debugNum; i++)
	{
	cout<<"idx : "<< i + debugPos <<" floatData : "<<debugFloatReverseData[i + debugPos]<<"\n";
	}
	free(debugFloatReverseData);
	//*/

	//lambda to k
	//*
	t1 = clock();
	double* gpuPXscale;
	int* gpuStepVal;
	gpuError = cudaMalloc((void**)&gpuPXscale, sizeof(double) * size_Z);
	gpuError = cudaMalloc((void**)&gpuStepVal, sizeof(int) * size_Z);

	gpuComputePXScale << <gridDimX, blockDimX >> > (gpuPXscale, size_Z);
	gpuError = cudaDeviceSynchronize();

	gpuComputeStepVal << <gridDimX, blockDimX >> > (gpuPXscale, gpuStepVal, size_Z);
	gpuError = cudaDeviceSynchronize();

	gpuStepValCheck << <gridDimX, blockDimX >> > (gpuStepVal, size_Z);
	gpuError = cudaDeviceSynchronize();
	//cout << "gpuError : " << cudaGetErrorString(gpuError) << "\n";

	/*
	int* StepVal = (int*)malloc(sizeof(int) * size_Z);
	gpuError = cudaMemcpy(StepVal, gpuStepVal, sizeof(int) * size_Z, cudaMemcpyDeviceToHost);
	double* PXscale = (double*)malloc(sizeof(double) * size_Z);
	gpuError = cudaMemcpy(PXscale, gpuPXscale, sizeof(double) * size_Z, cudaMemcpyDeviceToHost);
	for (int i = 1945; i < 1965; i++)// 1955 +- 10
	{
		cout << "i " << i << " " << StepVal[i] << "\n";
		//cout<<"i "<< i << " " << PXscale[i] << "\n";
	}
	//*/
	gpuFrequencyAdjust << <gridDimX, blockDimX >> > (gpuStepVal, gpuReversedData, size_Z, floatDataSize);//gpuCutDataZ
	gpuError = cudaDeviceSynchronize();
	//cout << "gpuError : " << cudaGetErrorString(gpuError) << "\n";
	cudaFree(gpuPXscale);
	cudaFree(gpuStepVal);
	t2 = clock();
	std::cout << "lambda to k done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";
	//*/

	// data out for RawDataScanP draw
	//*
	if (RawDataScanP != NULL)
		free(RawDataScanP);
	RawDataScanP = (float*)malloc(sizeof(float) * size_Z * size_Y);
	gpuError = cudaMemcpy(RawDataScanP, gpuReversedData, sizeof(float) * size_Z * size_Y, cudaMemcpyDeviceToHost);
	std::cout << "RawDataScanP[size_Z*125]"<< RawDataScanP[size_Z*125] << " \n";
	std::cout << "RawDataScanP[size_Z*126-1]" << RawDataScanP[size_Z * 126-1] << " \n";
	int over_count = 0;
	for (int i = 0; i < size_Z; i++)
	{
		if (abs(RawDataScanP[size_Z * 125 + i]) > 10000) {
			std::cout << "RawDataScanP[size_Z*125 + i" << i << "] : " << RawDataScanP[size_Z * 125 + i] << " \n";
			over_count++;
		}
	}
	std::cout << "over count:" << over_count << "\n";
	//*/
	
	
	// Sample X Y
	//*
	bool hadSample = sample_X != 1 || sample_Y != 1;
	float* gpuSampleData;
	if (hadSample)
	{
		t1 = clock();

		gpuError = cudaMalloc((void**)&gpuSampleData, sizeof(float) * floatDataSize);

		gpuSampleXY << <gridDimX, blockDimX >> >(gpuReversedData, gpuSampleData, floatDataSize, size_Y, size_Z, sample_X, sample_Y);
		gpuError = cudaDeviceSynchronize();

		floatDataSize = floatDataSize / (sample_X * sample_Y);
		size_X = size_X / sample_X;
		size_Y = size_Y / sample_Y;

		cudaFree(gpuReversedData);

		t2 = clock();
		//cout << "gpuError : " << cudaGetErrorString(gpuError) << "\n";
		std::cout << "floatDataSize : " << floatDataSize << "\n";
		std::cout << "SampleXY done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";
	}//*/

	// CUFFT
	t1 = clock();
	cufftHandle plan;
	cufftComplex *gpuComplexData;
	int NX = size_Z, BATCH = floatDataSize / size_Z;

	floatDataSize = floatDataSize / 2;
	size_Z = size_Z / 2;

	//split for reduce max gpu memory use
	BATCH = BATCH / split;

	gpuError = cudaMalloc((void**)&gpuFloatData, sizeof(float) * floatDataSize);
	for (int i = 0; i < split; i++)
	{
		// set cufftplan
		cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH);
		gpuError = cudaMalloc((void**)&gpuComplexData, sizeof(cufftComplex) * NX * BATCH);

		// data in  gpuCutDataZ
		if(hadSample)
			gpuDataToComplex << <gridDimX, blockDimX >> > (gpuSampleData, gpuComplexData, NX * BATCH, NX * BATCH * i);
		else
			gpuDataToComplex << <gridDimX, blockDimX >> > (gpuReversedData, gpuComplexData, NX * BATCH, NX * BATCH * i);
		gpuError = cudaDeviceSynchronize();

		// excute cufft
		cufftExecC2C(plan, gpuComplexData, gpuComplexData, CUFFT_FORWARD);
		gpuError = cudaDeviceSynchronize();

		// data out and cut mirror data
		gpuComplexToData << <gridDimX, blockDimX >> > (gpuComplexData, gpuFloatData, NX * BATCH / 2, size_Z, NX * BATCH / 2 * i);
		gpuError = cudaDeviceSynchronize();

		cufftDestroy(plan);
		cudaFree(gpuComplexData);
	}
	if (hadSample)
		cudaFree(gpuSampleData);
	else
		cudaFree(gpuReversedData);
	t2 = clock();
	std::cout << "cuff nx : " << NX << " Batch : " << BATCH << " split : " << split << "\n";
	std::cout << "CUFFT done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";

	/* debug output : CUFFT
	cout<<"gpuError : "<<cudaGetErrorString(gpuError)<<"\n";
	float* debugCUFFTData = (float*) malloc(sizeof(float) * floatDataSize);
	gpuError = cudaMemcpy(debugCUFFTData, gpuCutDataZ, sizeof(float) * floatDataSize, cudaMemcpyDeviceToHost);
	for(int i = 0; i < debugNum; i++)
	{
	cout<<"idx : "<< i + debugPos <<" floatData : "<<debugCUFFTData[i + debugPos]<<"\n";
	}
	free(debugCUFFTData);
	//*/

	// Shift data
	t1 = clock();
	float* gpuShiftData;
	gpuError = cudaMalloc((void**)&gpuShiftData, sizeof(float) * floatDataSize);

	gpuShift << <gridDimX, blockDimX >> >(gpuFloatData, gpuShiftData, floatDataSize, size_X, size_Y, size_Z, shift_X/sample_X);
	gpuError = cudaDeviceSynchronize();

	cudaFree(gpuFloatData);

	t2 = clock();
	//cout << "Shift data size : " << floatDataSize << "\n";
	std::cout << "Shift data done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";

	/* debug output : Shift data
	cout<<"gpuError : "<<cudaGetErrorString(gpuError)<<"\n";
	float* debugShiftData = (float*) malloc(sizeof(float) * floatDataSize);
	gpuError = cudaMemcpy(debugShiftData, gpuShiftData, sizeof(float) * floatDataSize, cudaMemcpyDeviceToHost);
	for(int i = 0; i < debugNum; i++)
	{
	cout<<"idx : "<< i + debugPos <<" floatData : "<<debugShiftData[i + debugPos]<<"\n";
	}
	free(debugShiftData);
	//*/

	// data out for opengl draw
	if (VolumeData != NULL)
		free(VolumeData);
	VolumeData = (float*)malloc(sizeof(float) * floatDataSize);
	gpuError = cudaMemcpy(VolumeData, gpuShiftData, sizeof(float) * floatDataSize, cudaMemcpyDeviceToHost);

	// Averge filter
	t1 = clock();
	float* gpuAvergeData;
	gpuError = cudaMalloc((void**)&gpuAvergeData, sizeof(float) *floatDataSize);

	gpuZaxisAverge << <gridDimX, blockDimX >> >(gpuShiftData, gpuAvergeData, floatDataSize, size_Z, avergeBlock);
	gpuError = cudaDeviceSynchronize();

	cudaFree(gpuShiftData);

	t2 = clock();
	//cout << "Averge size : " << floatDataSize << " avergeBlock : " << avergeBlock << "\n";
	std::cout << "Averge done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";

	/* debug output : Averge data
	cout<<"gpuError : "<<cudaGetErrorString(gpuError)<<"\n";
	float* debugFloatAvergeData = (float*) malloc(sizeof(float) * floatDataSize);
	gpuError = cudaMemcpy(debugFloatAvergeData, gpuAvergeData, sizeof(float) * floatDataSize, cudaMemcpyDeviceToHost);
	for(int i = 0; i < debugNum; i++)
	{
	cout<<"idx : "<< i + debugPos <<" floatAvergeData : "<<debugFloatAvergeData[i + debugPos]<<"\n";
	}
	free(debugFloatAvergeData);
	//*/

	if (VolumeDataAvg != NULL)
		free(VolumeDataAvg);
	VolumeDataAvg = (float*)malloc(sizeof(float) * floatDataSize);
	gpuError = cudaMemcpy(VolumeDataAvg, gpuAvergeData, sizeof(float) * floatDataSize, cudaMemcpyDeviceToHost);

	// Sum Thresh
	t1 = clock();

	gpuSumThresh << <gridDimX, blockDimX >> >(gpuAvergeData, floatDataSize, size_Z, depthFRange, depthBRange);
	gpuError = cudaDeviceSynchronize();

	t2 = clock();
	std::cout << "gpuSumThresh done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";

	/* debug output : Sum Thresh
	float* sumThresh = (float*)malloc(sizeof(float) * 1);
	gpuError = cudaMemcpy(sumThresh, gpuAvergeData, sizeof(float) * 1, cudaMemcpyDeviceToHost);
	cout << "Sum Thresh : " << *sumThresh << "\n";
	float* maxThresh = (float*)malloc(sizeof(float) * 1);
	gpuError = cudaMemcpy(maxThresh, gpuAvergeData + size_Z-1, sizeof(float) * 1, cudaMemcpyDeviceToHost);
	cout << "Max Thresh : " << *maxThresh << "\n";
	//*/

	// Peak Detect
	t1 = clock();
	int* gpuPointType;
	gpuError = cudaMalloc((void**)&gpuPointType, sizeof(int) *floatDataSize);

	gpuPeakDetect << <gridDimX, blockDimX >> >(gpuAvergeData, gpuPointType, floatDataSize, size_Z, peakGap, energyGap, depthFRange, depthBRange);
	gpuError = cudaDeviceSynchronize();

	cudaFree(gpuAvergeData);
	t2 = clock();
	//cout << "Peak Detect size : " << floatDataSize << " peakGap : " << peakGap << " energyGap : " << energyGap << " depthBRange : " << depthBRange << "\n";
	std::cout << "Peak Detect done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";

	/* debug output : Peak Detect
	cout<<"gpuError : "<<cudaGetErrorString(gpuError)<<"\n";
	int* debugPointType = (int*) malloc(sizeof(int) * floatDataSize);
	gpuError = cudaMemcpy(debugPointType, gpuPointType, sizeof(int) * floatDataSize, cudaMemcpyDeviceToHost);
	for(int i = 0; i < debugNum; i++)
	{
	cout<<"idx : "<<(i + debugPos)*size_Z <<" PointType : "<<debugPointType[(i + debugPos)*size_Z]<<"\n";
	}
	free(debugPointType);
	//*/
	// Min Peak
	t1 = clock();
	int* gpuMinPeak;
	gpuError = cudaMalloc((void**)&gpuMinPeak, sizeof(int) *size_X);

	gpuFindMinPeak << <gridDimX, blockDimX >> >(gpuPointType, gpuMinPeak, size_X, size_Y, size_Z);
	gpuError = cudaDeviceSynchronize();

	gpuFindMinPeakOne << <gridDimX, blockDimX >> >(gpuMinPeak, size_X);
	gpuError = cudaDeviceSynchronize();

	int tmpMinPeak = size_Z;
	gpuError = cudaMemcpy(&tmpMinPeak, gpuMinPeak, sizeof(int) * 1, cudaMemcpyDeviceToHost);
	/*
	int* MinPeak = (int*)malloc(sizeof(int) * size_X);
	gpuError = cudaMemcpy(MinPeak, gpuMinPeak, sizeof(int) * size_X, cudaMemcpyDeviceToHost);
	for (int i = 0; i < size_X; i++)
	{
		if (MinPeak[i] < tmpMinPeak && MinPeak[i] != 0)
			tmpMinPeak = MinPeak[i];
	}
	free(MinPeak);
	//*/
	cudaFree(gpuMinPeak);

	// Set MinPeak
	gpuSetMinPeak << <gridDimX, blockDimX >> >(gpuPointType, floatDataSize, size_X, size_Y, size_Z, tmpMinPeak, boardNRange);
	gpuError = cudaDeviceSynchronize();
	t2 = clock();
	//cout << "Min Peak size : " << size_X << " Min Peak : " << tmpMinPeak << "\n";
	std::cout << "Min Peak done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";
	/* debug output : Min Peak
	cout<<"gpuError : "<<cudaGetErrorString(gpuError)<<"\n";
	for(int i = 0; i < debugNum; i++)
	{
	if(i + debugPos >= size_X)
	break;
	cout<<"idx : "<<(i + debugPos) <<" PointType : "<<MinPeak[(i + debugPos)]<<"\n";
	}
	//*/
	// Board Detect
	t1 = clock();
	for (int i = 0; i < 40; i++) // i < max(size_X, size_Y) * 2
	{
		gpuBoardDetect << <gridDimX, blockDimX >> >(gpuPointType, floatDataSize, size_X, size_Y, size_Z, depthFRange, depthBRange, boardNRange);
		gpuError = cudaDeviceSynchronize();
	}

	t2 = clock();
	//cout << "Board Detect size : " << floatDataSize << " max(size_X, size_Y)*2 : " << max(size_X, size_Y) * 2 << " boardNRange : " << boardNRange << "\n";
	std::cout << "Board Detect done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";

	/* debug output : Board Detect
	cout<<"gpuError : "<<cudaGetErrorString(gpuError)<<"\n";
	int* debugBoardTemp = (int*) malloc(sizeof(int) * floatDataSize);
	gpuError = cudaMemcpy(debugBoardTemp, gpuPointType, sizeof(int) * floatDataSize, cudaMemcpyDeviceToHost);
	for(int i = 0; i < debugNum; i++)
	{
		cout<<"idx : "<<(i + debugPos)*size_Z <<" PointType : "<<debugBoardTemp[(i + debugPos)*size_Z]<<" idx1 : "<<debugBoardTemp[(i + debugPos)*size_Z + 1]<<"\n";
		cout << "idx5 : " << debugBoardTemp[(i + debugPos)*size_Z + 5] << " idx6 : " << debugBoardTemp[(i + debugPos)*size_Z + 6] << " idx7 : " << debugBoardTemp[(i + debugPos)*size_Z + 7] << " ";
		cout << "idx8 : " << debugBoardTemp[(i + debugPos)*size_Z + 8] << " idx9 : " << debugBoardTemp[(i + debugPos)*size_Z + 9] <<  " idx4 : " << debugBoardTemp[(i + debugPos)*size_Z + 4] << "\n";
	}
	free(debugBoardTemp);
	//*/

	//Test Mapping Time
	//*
	t1 = clock();
	int* gpuMappingT;
	gpuError = cudaMalloc((void**)&gpuMappingT, sizeof(int) *floatDataSize);

	gpuMapping << <gridDimX, blockDimX >> >(gpuPointType, gpuMappingT, size_X, size_Y, size_Z);
	gpuError = cudaDeviceSynchronize();
	cudaFree(gpuMappingT);

	t2 = clock();
	std::cout << "Mapping done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";

	//*/
	//pull data from gpu to cpu
	t1 = clock();
	if (PointType != NULL)
		free(PointType);
	PointType = (int*)malloc(sizeof(int) * floatDataSize);
	gpuError = cudaMemcpy(PointType, gpuPointType, sizeof(int) * floatDataSize, cudaMemcpyDeviceToHost);

	cudaFree(gpuPointType);

	VolumeSize_X = size_X;
	VolumeSize_Y = size_Y;
	VolumeSize_Z = size_Z;
	std::cout << "X: " << size_X << " Y: " << size_Y << " Z: " << size_Z << "\n";
	t2 = clock();
	std::cout << "Pull Data done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s\n";
	t4 = clock();
	std::cout << "RawToPointCloud Total t: " << (t4 - t3) / (double)(CLOCKS_PER_SEC) << " s\n";
}

void TRcuda::Test()
{

}
