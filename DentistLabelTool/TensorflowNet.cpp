#include "TensorflowNet.h"

TensorflowNet::TensorflowNet()
{
	//PythonModule
	module = new PythonModule("TensorflowNet.OtherSide.TensorflowNet_C");
	module->SetArgs(1);
}
TensorflowNet::~TensorflowNet()
{
}

double** TensorflowNet::Predict(double** img)
{
	int OutRows, OutCols;
	module->AddArgs(img, 250, 250, 0);
	return module->CallFunction_ReturnNumpy2DArray("PredictImg", OutRows, OutCols);
}
