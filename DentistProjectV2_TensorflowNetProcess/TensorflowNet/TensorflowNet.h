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
	float**		Predict_OtherSide(float**);
	void		Predict_Full(int, int, string);
	void		DeleteArray(float**);

private:
	PythonModule<float>* module = NULL;
};

// 由於這個網址的關係，GPU 的部分跟著 process，所以如果創建之後，只有在 Process 關掉才可以施放 GPU
// https://stackoverflow.com/questions/39758094/clearing-tensorflow-gpu-memory-after-model-execution