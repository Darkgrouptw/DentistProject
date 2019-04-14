#include "PythonModule.h"

PythonModule::PythonModule(string ModuleName)
{
	Init(ModuleName.c_str());
}
PythonModule::~PythonModule()
{
	Close();
}

// 傳送 Function 會用到的
void PythonModule::SetArgs(int size)
{
	// 清空
	SAVE_DELETE_PY(pyArgs);

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

// Return 的部分
void PythonModule::CallFunction(string FunctionName)
{
	// 拿取 Function
	PyObject* pyFunc = PyObject_GetAttrString(pyModule, FunctionName.c_str());
	GetPythonError(pyFunc);

	// 傳進 Function
	PyEval_CallObject(pyFunc, pyArgs);

	//清空
	SAVE_DELETE_PY(pyFunc);
}
int PythonModule::CallFunction_ReturnInt(string FunctionName)
{
	// 拿取 Function
	PyObject* pyFunc = PyObject_GetAttrString(pyModule, FunctionName.c_str());
	GetPythonError(pyFunc);

	// 傳進 Function
	PyObject* pyReturn = NULL;
	pyReturn = PyEval_CallObject(pyFunc, pyArgs);
	GetPythonError(pyReturn);

	// 取值
	int result = -1;
	PyArg_Parse(pyReturn, "i", &result);

	//清空
	SAVE_DELETE_PY(pyReturn);
	SAVE_DELETE_PY(pyFunc);

	return result;
}

// Python 連接的函數
void PythonModule::Init(const char* ModuleName)
{
	// 初始化
	Py_Initialize();

	// 抓那個 Module
	PyObject* pyCode = NULL;
	pyCode = PyUnicode_FromString(ModuleName);
	GetPythonError(pyCode);

	// Import
	pyModule = PyImport_Import(pyCode);
	GetPythonError(pyModule);
	Py_DECREF(pyCode);
}
void PythonModule::Close()
{
	SAVE_DELETE_PY(pyModule);
	SAVE_DELETE_PY(pyArgs);
}

// Helper Function
void PythonModule::GetPythonError(PyObject* pointer)
{
	if (pointer == NULL)
	{
		PyErr_Print();
		assert(false);
	}
}