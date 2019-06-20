#include "CalibrationUtility.h"
#include "PythonModule.cpp"

CalibrationUtility::CalibrationUtility()
{
	module = new PythonModule<float>("Calibration.Calibration_CPP");
}
CalibrationUtility::~CalibrationUtility()
{
}

// 校正工具相關
float** CalibrationUtility::Calibrate(float** data, int rows, int cols)
{
	int OutRows, OutCols;

	module->SetArgs(1);
	module->AddArgs(data, rows, cols, 0);
	return module->CallFunction_ReturnNumpy2DArray("CalibrationAPI", OutRows, OutCols);
}
void CalibrationUtility::DeleteArray(float** arrayData)
{
	module->Delete2DArray(arrayData);
}
