/*
加速用的工具
*/
#include "GlobalDefine.h"

#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <cassert>
#include <numeric>
#include <algorithm>

#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define generic __identifier(generic)
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#undef generic

using namespace std;
using namespace cv;

typedef unsigned short ushort;
typedef unsigned char uchar;

class UtilityTools
{
public:
	UtilityTools();
	~UtilityTools();

};