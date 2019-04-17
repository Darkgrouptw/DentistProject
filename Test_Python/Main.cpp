#include <iostream>
#include <Windows.h>
#include "PythonModule.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////
// 主要 Function
//////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
	PythonModule module("TensorflowTestPY");

	cout << "==============================" << endl;

	module.CallFunction("TestPrint");

	cout << "==============================" << endl;

	module.SetArgs(2);
	module.AddArgs(10, 0);
	module.AddArgs(28, 1);
	cout << "C++ add: " << module.CallFunction_ReturnInt("AddTest") << endl;

	cout << "==============================" << endl;

	module.SetArgs(2);
	module.AddArgs(10, 0);
	module.AddArgs(28, 1);
	module.CallFunction("TensorTest");

	cout << "==============================" << endl;
	//double CArrays[4][3] = { { 1.3, 2.4, 5.6 },{ 4.5, 7.8, 8.9 },{ 1.7, 0.4, 0.8 }, {8, 9, 10} };
	double Array1D[5] = { 1, 2, 3, 4, 6.55 };
	module.SetArgs(1);
	module.AddArgs(Array1D, 5, 0);
	module.CallFunction("NumpyArrayTest");

	cout << "==============================" << endl;
	double CArrays[4][3] = { { 1.3, 2.4, 5.6 },{ 4.5, 7.8, 8.9 },{ 1.7, 0.4, 0.8 }, {8, 9, 10} };
	module.SetArgs(1);
	module.AddArgs((double**)CArrays, 4, 3, 0);
	module.CallFunction("NumpyArrayTest");

	cout << "==============================" << endl;
	module.SetArgs(1);
	module.AddArgs((double**)CArrays, 4, 3, 0);
	int rows, cols;
	double** NumpyArray = module.CallFunction_ReturnNumpy2DArray("NumpyOperationTest", rows, cols);
	// 測試
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			cout << NumpyArray[i][j] << " ";
		cout << endl;
	}
	module.Delete2DArray(NumpyArray);

	//cout << T
	system("PAUSE");
	return 0;
}