#include "PythonModule.h"

PythonModule::PythonModule(string ModuleName)
{
	Init(ModuleName.c_str());
}
PythonModule::~PythonModule()
{
	cout << "Delete PyModule" << endl;
	Close();
}

// 傳送 Function 會用到的
void PythonModule::SetArgs(int size)
{
	// 清空
	Py_XDECREF(pyArgs);

	// 新增大小
	pyArgs = PyTuple_New(size);
	ArgSize = size;
}
void PythonModule::AddArgs(int value, int index)
{
	assert(pyArgs != NULL && "必須初始化參數!!");
	if (index >= ArgSize)
		assert(false && "必須要大於 初始化的 Size");
	PyTuple_SET_ITEM(pyArgs, index, Py_BuildValue("i", value));
}
void PythonModule::AddArgs(double value, int index)
{
	assert(pyArgs != NULL && "必須初始化參數!!");
	if (index >= ArgSize)
		assert(false && "必須要大於 初始化的 Size");
	PyTuple_SET_ITEM(pyArgs, index, Py_BuildValue("d", value));
}
void PythonModule::AddArgs(double* value, int size, int index)
{
	// 設定大小 & 值
	npy_intp Dims[1] = { size };
	PyObject* pyArray = PyArray_SimpleNewFromData(1, Dims, NPY_DOUBLE, value);
	GetPythonError();

	PyTuple_SET_ITEM(pyArgs, index, pyArray);
}
void PythonModule::AddArgs(double** value, int rows, int cols, int index)
{
	// 設定大小 & 值
	npy_intp Dims[2] = { rows, cols };
	PyObject* pyArray = PyArray_SimpleNewFromData(2, Dims, NPY_DOUBLE, value);
	GetPythonError();

	PyTuple_SET_ITEM(pyArgs, index, pyArray);
}

// Return 的部分
void PythonModule::CallFunction(string FunctionName)
{
	// 拿取 Function
	PyObject* pyFunc = PyObject_GetAttrString(pyModule, FunctionName.c_str());
	GetPythonError();
	PyErr_Print();

	// 傳進 Function
	PyEval_CallObject(pyFunc, pyArgs);

	//清空
	Py_XDECREF(pyFunc);
	Py_XDECREF(pyArgs);

	// Last Check
	PyErr_Print();
}
int PythonModule::CallFunction_ReturnInt(string FunctionName)
{
	// 拿取 Function
	PyObject* pyFunc = PyObject_GetAttrString(pyModule, FunctionName.c_str());
	GetPythonError();

	// 傳進 Function
	PyObject* pyReturn = NULL;
	pyReturn = PyEval_CallObject(pyFunc, pyArgs);
	GetPythonError();


	// 取值
	int result = -1;
	PyArg_Parse(pyReturn, "i", &result);
	PyErr_Print();

	//清空
	Py_XDECREF(pyReturn);
	Py_XDECREF(pyFunc);
	Py_XDECREF(pyArgs);

	// Last Check
	PyErr_Print();

	return result;
}
double** PythonModule::CallFunction_ReturnNumpy2DArray(string FunctionName, int& OutRows, int& OutCols)
{
	// 拿取 Function
	PyObject* pyFunc = PyObject_GetAttrString(pyModule, FunctionName.c_str());
	GetPythonError();
	if (!PyCallable_Check(pyFunc))
		assert(false && "此 Function 不能使用 or 設定錯誤!!");

	// 傳進 Function
	PyObject* pyReturn = NULL;
	pyReturn = PyEval_CallObject(pyFunc, pyArgs);
	GetPythonError();

	// 取值
	npy_intp* Dims = NULL;
	double** NumpyObject = NULL;
	double* _NumpyList = NULL;

	// 抓取資料
	PyArrayObject* npObject = (PyArrayObject*)pyReturn;
	assert(npObject->nd == 2 && "Array 大小不符!!");
	cout << "Return To C: " << npObject->dimensions[0] << " " << npObject->dimensions[1] << " ND:" << npObject->nd << endl;

	// 把資料轉成 1D
	int Size1D = npObject->dimensions[0] * npObject->dimensions[1];
	_NumpyList = new double[Size1D];
	memset(_NumpyList, 0, sizeof(double) * Size1D);
	memcpy(_NumpyList, npObject->data, sizeof(double) * Size1D);

	// 再到2D
	NumpyObject = new double*[npObject->dimensions[0]];
	for (int i = 0; i < npObject->dimensions[0]; i++)
		NumpyObject[i] = &_NumpyList[i * npObject->dimensions[1]];

	// 設定傳送到外的參數
	OutRows = npObject->dimensions[0];
	OutCols = npObject->dimensions[1];

	PyErr_Print();

	//清空
	Py_XDECREF(pyReturn);
	Py_XDECREF(pyFunc);
	Py_XDECREF(pyArgs);

	// Last Check
	PyErr_Print();

	return NumpyObject;
}

// 清除部分
void PythonModule::Delete2DArray(double** Array)
{
	cout << "Delete: " << Array[0] << " " << &Array[0][0] << endl;
	delete[] &Array[0][0];
	delete[] Array;
}

// Python 連接的函數
void PythonModule::Init(const char* ModuleName)
{
	// 初始化
	Py_Initialize();
	InitNumpy();

	// 抓那個 Module
	PyObject* pyCode = NULL;
	pyCode = PyUnicode_FromString(ModuleName);
	GetPythonError();

	// Import
	pyModule = PyImport_Import(pyCode);
	GetPythonError();
	Py_DECREF(pyCode);
}
void PythonModule::Close()
{
	Py_DECREF(pyModule);	
	Py_XDECREF(pyArgs);						// 差別只在 XDECREF 不會刪除 NULL

	// 關閉 Python Interpreter
	Py_Finalize();
}

// Helper Function
void PythonModule::GetPythonError()
{
	if (PyErr_Occurred())
	{
		PyErr_Print();
		assert(false);
	}
}
int PythonModule::InitNumpy()
{
	import_array();
}