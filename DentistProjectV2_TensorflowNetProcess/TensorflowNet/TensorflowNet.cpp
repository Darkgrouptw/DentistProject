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
void TensorflowNet::Predict_Full(int StartIndex, int EndIndex, string Path)
{
	module->SetArgs(3);
	module->AddArgs(StartIndex, 0);
	module->AddArgs(EndIndex, 1);
	module->AddArgs(Path, 2);
	module->CallFunction("PredictImg_Full");
}
void TensorflowNet::DeleteArray(float** array)
{
	module->Delete2DArray(array);
}