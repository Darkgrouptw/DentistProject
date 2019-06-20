#pragma once
#include "PythonModule.h"

class CalibrationUtility
{
public:
	CalibrationUtility();
	~CalibrationUtility();

	//////////////////////////////////////////////////////////////////////////
	// 校正工具相關
	//////////////////////////////////////////////////////////////////////////
	float**			Calibrate(float**, int, int);
	void			DeleteArray(float**);

private:
	PythonModule<float>* module = NULL;
};

