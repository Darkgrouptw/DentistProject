#pragma once
#include "PythonModule.h"

class TensorflowNet_OtherSide
{
public:
	TensorflowNet_OtherSide();
	~TensorflowNet_OtherSide();

	//////////////////////////////////////////////////////////////////////////
	// 網路相關
	//////////////////////////////////////////////////////////////////////////
	float**		Predict(float**);
	void		DeleteArray(float**);

private:
	PythonModule<float>* module = NULL;
};