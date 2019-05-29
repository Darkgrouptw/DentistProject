#include "TensorflowNet_OtherSide.h"
#include "PythonModule.cpp"

TensorflowNet_OtherSide::TensorflowNet_OtherSide()
{
	//PythonModule
	module = new PythonModule<float>("TensorflowNet.OtherSide.TensorflowNet_C");
}
TensorflowNet_OtherSide::~TensorflowNet_OtherSide()
{
}

// 網路相關
float** TensorflowNet_OtherSide::Predict(float** img)
{
	int OutRows, OutCols;
	module->SetArgs(1);
	module->AddArgs(img, 250, 250, 0);
	return module->CallFunction_ReturnNumpy2DArray("PredictImg", OutRows, OutCols);
}
void TensorflowNet_OtherSide::DeleteArray(float** array)
{
	module->Delete2DArray(array);
}
