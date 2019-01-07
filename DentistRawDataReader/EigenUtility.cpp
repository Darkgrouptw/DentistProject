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
	MatrixXf X = EigenMatrixA.ldlt().solve(EigenMatrixB);
	params = X.data();
}