#include "EigenUtility.h"
#include <Eigen/Dense>

using namespace Eigen;

typedef Matrix<float, -1, -1, Eigen::RowMajor> MatrixXRf;

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
	// 設定原本的變數
	this->NumPolynomial = NumPolynomial;

	MatrixXRf EigenMatrixA = Map<MatrixXRf>(MatrixA, NumPolynomial + 1, NumPolynomial + 1);
	MatrixXRf EigenMatrixB = Map<MatrixXRf>(MatrixB, NumPolynomial + 1, 1);
	MatrixXRf X = EigenMatrixA.ldlt().solve(EigenMatrixB);
	params = X.data();
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