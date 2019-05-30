#pragma once
#include <iostream>
#include <cassert>
#include <string>

#include <Python.h>
#include <numpy/arrayobject.h>

using namespace std;

class PythonModule
{
public:
	PythonModule(string);
	~PythonModule();

	//////////////////////////////////////////////////////////////////////////
	// 傳送 Function 會用到的
	//
	// 注意事項
	// 1. 如果發現 Print不出東西，那很有可能 Python 有 Error 訊息
	// 2. 最好每一個 Function 後面都檢查看有沒有 Error 訊息，因為不一定回傳的值是 NULL 無法透過 GetPythonError 抓出結果
	//////////////////////////////////////////////////////////////////////////
	void SetArgs(int);
	void AddArgs(int, int);
	void AddArgs(double, int);
	void AddArgs(double*, int, int);											// 使用完的值記得刪掉
	void AddArgs(double**, int, int, int);										// 同上

	//////////////////////////////////////////////////////////////////////////
	// Return 的部分 
	// 
	// 要注意事項:
	// 1. 這邊有一個問題，就是如果少傳參數會報錯
	//////////////////////////////////////////////////////////////////////////
	void CallFunction(string);
	int CallFunction_ReturnInt(string);
	double** CallFunction_ReturnNumpy2DArray(string, int&, int&);				// 要用這個 API，如果以後要改成其他維度的話 https://docs.scipy.org/doc/numpy/reference/c-api.array.html#data-access

	//////////////////////////////////////////////////////////////////////////
	// 清除變數的 Function
	//////////////////////////////////////////////////////////////////////////
	void Delete2DArray(double **);												// 清除陣列

private:
	//////////////////////////////////////////////////////////////////////////
	// Python 連接的函數
	//////////////////////////////////////////////////////////////////////////
	void Init(const char*);
	void Close();
	PyObject* pyModule = NULL;
	PyObject* pyArgs = NULL;
	int ArgSize = -1;

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	void GetPythonError();
	int InitNumpy();
};

// 注意事項：請勿重複使用兩次，有 bug
// 如果要使用兩次，就把 Py_Finalize 註解，應該就沒問題了吧@@
// https://docs.python.org/2/c-api/init.html#c.Py_Finalize
//
// Python		C++
// float32	=>	float
// float64	=>	double
// https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html