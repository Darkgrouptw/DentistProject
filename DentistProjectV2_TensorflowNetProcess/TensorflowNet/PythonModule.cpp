#include "PythonModule.h"

template <typename T> PythonModule<T>::PythonModule(string ModuleName)
{
	Init(ModuleName.c_str());
}
template <typename T> PythonModule<T>::~PythonModule()
{
	Close();
}

// 傳送 Function 會用到的
template <typename T> void PythonModule<T>::SetArgs(int size)
{
	// 清空
	Py_XDECREF(pyArgs);

	// 新增大小
	pyArgs = PyTuple_New(size);
	ArgSize = size;
}
template <typename T> void PythonModule<T>::AddArgs(int value, int index)
{
	assert(pyArgs != NULL && "必須初始化參數!!");
	if (index >= ArgSize)
		assert(false && "必須要大於 初始化的 Size");
	PyTuple_SetItem(pyArgs, index, Py_BuildValue("i", value));
	//PyTuple_SetItem()
}
template <typename T> void PythonModule<T>::AddArgs(string value, int index)
{
	assert(pyArgs != NULL && "必須初始化參數!!");
	if (index >= ArgSize)
		assert(false && "必須要大於 初始化的 Size");
	PyTuple_SetItem(pyArgs, index, Py_BuildValue("s", value.c_str()));
}
template <typename T> void PythonModule<T>::AddArgs(T value, int index)
{
	assert(pyArgs != NULL && "必須初始化參數!!");
	if (index >= ArgSize)
		assert(false && "必須要大於 初始化的 Size");
	PyTuple_SetItem(pyArgs, index, Py_BuildValue("d", value));
}
template <typename T> void PythonModule<T>::AddArgs(T* value, int size, int index)
{

	// 設定大小 & 值
	npy_intp Dims[1] = { size };
	PyObject* pyArray = PyArray_SimpleNewFromData(1, Dims, NumpyTypeNumber(), value);
	GetPythonError();

	PyTuple_SetItem(pyArgs, index, pyArray);
}
template <typename T> void PythonModule<T>::AddArgs(T** value, int rows, int cols, int index)
{
	// 這個方法由於 Python 創建 二維的大小是不連續的
	// 所以不能使用這個方法
	// https://stackoverflow.com/questions/27940848/passing-2-dimensional-c-array-to-python-numpy
	/*PyObject* pyArray = PyArray_SimpleNewFromData(2, Dims, NPY_DOUBLE, value);
	GetPythonError(pyArray);*/
	
	// 這邊要先做一個型別判斷

	// 設定大小 & 值
	npy_intp Dims[2] = { rows, cols };
	PyObject* pyArray = PyArray_SimpleNew(2, Dims, NumpyTypeNumber());

	T *p = (T *)PyArray_DATA(pyArray);
	for (int k = 0; k < rows; ++k) {
		memcpy(p, value[k], sizeof(T) * cols);
		p += cols;
	}

	PyTuple_SetItem(pyArgs, index, pyArray);
}

// Return 的部分
template <typename T> void PythonModule<T>::CallFunction(string FunctionName)
{
	// 拿取 Function
	PyObject* pyFunc = PyObject_GetAttrString(pyModule, FunctionName.c_str());
	GetPythonError();
	if (!PyCallable_Check(pyFunc))
		assert(false && "此 Function 不能使用 or 設定錯誤!!");

	// 傳進 Function
	PyEval_CallObject(pyFunc, pyArgs);

	//清空
	Py_DECREF(pyFunc);
	Py_XDECREF(pyArgs);

	// Last Check
	PyErr_Print();
}
template <typename T> int PythonModule<T>::CallFunction_ReturnInt(string FunctionName)
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
	int result = -1;
	PyArg_Parse(pyReturn, "i", &result);
	PyErr_Print();

	//清空
	Py_XDECREF(pyReturn);
	Py_DECREF(pyFunc);
	Py_XDECREF(pyArgs);

	// Last Check
	PyErr_Print();

	return result;
}
template <typename T> T** PythonModule<T>::CallFunction_ReturnNumpy2DArray(string FunctionName, int& OutRows, int& OutCols)
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
	// 根據底下的連結可以知道
	// Python		C++
	// float32	=>	float
	// float64	=>	double
	// https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
	npy_intp* Dims = NULL;
	T** NumpyObject = NULL;
	T* _NumpyList = NULL;

	// 抓取資料
	PyArrayObject* npObject = (PyArrayObject*)pyReturn;
	assert(pyReturn != NULL && "Return 資料是空的!!");
	assert(npObject->nd == 2 && "Array 大小不符!!");

	// 把資料轉成 1D
	int Size1D = npObject->dimensions[0] * npObject->dimensions[1];
	_NumpyList = new T[Size1D];
	memset(_NumpyList, 0, sizeof(T) * Size1D);
	memcpy(_NumpyList, npObject->data, sizeof(T) * Size1D);

	// 再到2D
	NumpyObject = new T*[npObject->dimensions[0]];
	for (int i = 0; i < npObject->dimensions[0]; i++)
		NumpyObject[i] = &_NumpyList[i * npObject->dimensions[1]];

	// 設定傳送到外的參數
	OutRows = npObject->dimensions[0];
	OutCols = npObject->dimensions[1];

	PyErr_Print();

	//清空
	Py_XDECREF(pyReturn);
	Py_DECREF(pyFunc);
	Py_XDECREF(pyArgs);

	// Last Check
	PyErr_Print();

	return NumpyObject;
}

// 清除部分
template <typename T> void PythonModule<T>::Delete2DArray(T** Array)
{
	if (Array != NULL)
	{
		delete[] & Array[0][0];
		delete[] Array;
	}
}

// Python 連接的函數
template <typename T> void PythonModule<T>::Init(const char* ModuleName)
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
template <typename T> void PythonModule<T>::Close()
{
	Py_DECREF(pyModule);
	Py_XDECREF(pyArgs);						// 差別只在 XDECREF 不會刪除 NULL

	// 關閉 Python Interpreter
	Py_Finalize();
}

// Helper Function
template <typename T> void PythonModule<T>::GetPythonError()
{
	if (PyErr_Occurred())
	{
		PyErr_Print();
		assert(false);
	}
}
template <typename T> int PythonModule<T>::InitNumpy()
{
	import_array();
}
template <typename T> int PythonModule<T>::NumpyTypeNumber()
{
	if (std::is_same<T, float>::value)
		return NPY_FLOAT;
	else if (std::is_same<T, double>::value)
		return NPY_DOUBLE;
	else
		assert(false && "請加入的型態!!");
}