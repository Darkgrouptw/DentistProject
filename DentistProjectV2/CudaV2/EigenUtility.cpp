#include "EigenUtility.h"
#include <Eigen/Dense>

using namespace Eigen;

EigenUtility::EigenUtility()
{
}
EigenUtility::~EigenUtility()
{
}

// 外部呼叫
void EigenUtility::SetAverageValue(float average)
{
	avg = average;
}
void EigenUtility::SolveByEigen(float* MatrixA, float* MatrixB, int NumPolynomial)
{
	MatrixXf EigenMatrixA = Map<MatrixXf>(MatrixA, NumPolynomial + 1, NumPolynomial + 1);
	MatrixXf EigenMatrixB = Map<MatrixXf>(MatrixB, NumPolynomial + 1, 1);
	MatrixXf X = EigenMatrixA.householderQr().solve(EigenMatrixB);
	params = X.data();

	// 設定原本的變數
	this->NumPolynomial = NumPolynomial;
}
float* EigenUtility::GetFunctionArray(int SizeZ, int YAverage)
{
	// 一定要確定他有改過
	assert(NumPolynomial != -1);

	float* Value = new float[SizeZ];
	for (int i = 0; i < SizeZ; i++)
	{
		float x = (float)i / SizeZ;
		float FunctionValue = 0;
		for (int j = 0; j <= NumPolynomial; j++)
			FunctionValue += params[j] * pow(x, NumPolynomial - j);
		Value[i] = FunctionValue  + YAverage;
	}
	return Value;
}