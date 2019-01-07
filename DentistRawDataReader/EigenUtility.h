#pragma once
#include <iostream>

using namespace std;

class EigenUtility
{
public:
	EigenUtility();
	~EigenUtility();

	// 外部呼叫
	void SetAverageValue(float);
	void SolveByEigen(float*, float*, int);

	float *params;					// 參數式多少

private:
	//////////////////////////////////////////////////////////////////////////
	// Function 一些參數
	//////////////////////////////////////////////////////////////////////////
	float avg;						// Y 的平均
	int NumPolynomial;				// 有幾次項
	
};

