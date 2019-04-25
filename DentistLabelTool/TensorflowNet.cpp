#include "TensorflowNet.h"
#include "PythonModule.cpp"

TensorflowNet::TensorflowNet()
{
	//PythonModule
	module = new PythonModule<float>("TensorflowNet.OtherSide.TensorflowNet_C");
}
TensorflowNet::~TensorflowNet()
{
}

// 網路相關
float** TensorflowNet::Predict(float** img)
{
	int OutRows, OutCols;
	module->SetArgs(1);
	module->AddArgs(img, 250, 250, 0);
	return module->CallFunction_ReturnNumpy2DArray("PredictImg", OutRows, OutCols);
}
void TensorflowNet::DeleteArray(float** array)
{
	module->Delete2DArray(array);
}
