#include <iostream>
#include <Windows.h>
#include "PythonModule.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////
// 主要 Function
//////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
	// 初始化
	//Py_Initialize();
	//PyObject* TestCodePY = PyUnicode_FromString("TensorflowTestPY");

	//// Load the module object
	//PyObject* pModule = PyImport_Import(TestCodePY);
	////GetPythonError(pModule);
	//Py_DECREF(TestCodePY);

	////////////////////////////////////////////////////////////////////////////
	//// 傳 0 個參數
	////////////////////////////////////////////////////////////////////////////
	//PyObject* pFunc1 = PyObject_GetAttrString(pModule, "TestPrint");
	////GetPythonError(pFunc1);
	//PyEval_CallObject(pFunc1, NULL);
	//Py_DECREF(pFunc1);
	//
	////////////////////////////////////////////////////////////////////////////
	//// 加法
	////////////////////////////////////////////////////////////////////////////
	//PyObject* pFunc2 = PyObject_GetAttrString(pModule, "AddTest");
	////GetPythonError(pFunc2);

	//PyObject* pArgs = PyTuple_New(2);
	//PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", 10));
	//PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", 30));

	//PyObject* pFunc2Return = PyEval_CallObject(pFunc2, pArgs);
	//int result = 100;
	//PyArg_Parse(pFunc2Return, "i", &result);
	//cout << "Python Return Value: " << result << endl;
	//Py_DECREF(pFunc2);
	//Py_DECREF(pArgs);
	//Py_DECREF(pFunc2Return);

	////////////////////////////////////////////////////////////////////////////
	//// Tensorflow 的加法
	////////////////////////////////////////////////////////////////////////////
	//PyObject* pyFunc3 = PyObject_GetAttrString(pModule, "TensorTest");
	////GetPythonError(pyFunc3);

	//pArgs = PyTuple_New(2);
	//PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", 28));
	//PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", 17));

	//PyEval_CallObject(pyFunc3, pArgs);
	//Py_DECREF(pFunc2);
	//Py_DECREF(pArgs);
	//Py_DECREF(pFunc2Return);


	//// 清除東西
	//Py_DECREF(pModule);

	//// Finish the Python Interpreter
	//Py_Finalize();


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
	module.CallFunction_ReturnInt("TensorTest");

	//cout << T
	system("PAUSE");
	return 0;
}