#pragma once
#include "PythonModule.h"

class TensorflowNet
{
public:
	TensorflowNet();
	~TensorflowNet();

	//////////////////////////////////////////////////////////////////////////
	// 網路相關
	//////////////////////////////////////////////////////////////////////////
	float**		Predict(float**);
	void		DeleteArray(float**);

private:
	PythonModule<float>* module = NULL;
};