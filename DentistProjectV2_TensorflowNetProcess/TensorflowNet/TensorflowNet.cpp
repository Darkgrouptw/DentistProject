#include "TensorflowNet.h"
#include "PythonModule.cpp"

TensorflowNet::TensorflowNet()
{
	//PythonModule
	module = new PythonModule<float>("TensorflowNet_C.TensorflowNet_C");
}
TensorflowNet::~TensorflowNet()
{
	delete module;
}

// 網路相關
float** TensorflowNet::Predict_OtherSide(float** img)
{
	int OutRows, OutCols;
	module->SetArgs(1);
	module->AddArgs(img, 250, 250, 0);
	return module->CallFunction_ReturnNumpy2DArray("PredictImg_OtherSide", OutRows, OutCols);
}
void TensorflowNet::DeleteArray(float** array)
{
	module->Delete2DArray(array);
}