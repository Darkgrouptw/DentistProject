#pragma once
#include "PythonModule.h"

class TensorflowNet
{
public:
	TensorflowNet();
	~TensorflowNet();

	double** Predict(double**);

private:
	PythonModule* module = NULL;
};

