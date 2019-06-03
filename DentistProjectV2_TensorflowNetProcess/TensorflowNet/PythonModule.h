#pragma once
#include <iostream>
#include <cassert>
#include <string>
#include <type_traits>

// 避免跟 QT 關鍵字衝突
#pragma push_macro("slots")
#undef slots
#include <Python.h>
#include <numpy/arrayobject.h>
#pragma pop_macro("slots")

using namespace std;

template <typename T>
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
	void AddArgs(string value, int index);
	void AddArgs(T, int);
	void AddArgs(T*, int, int);													// 使用完的值記得刪掉
	void AddArgs(T**, int, int, int);											// 同上

	//////////////////////////////////////////////////////////////////////////
	// Return 的部分 
	// 
	// 要注意事項:
	// 1. 這邊有一個問題，就是如果少傳參數會報錯
	//////////////////////////////////////////////////////////////////////////
	void CallFunction(string);
	int CallFunction_ReturnInt(string);
	T** CallFunction_ReturnNumpy2DArray(string, int&, int&);					// 要用這個 API，如果以後要改成其他維度的話

	//////////////////////////////////////////////////////////////////////////
	// 清除變數的 Function
	//////////////////////////////////////////////////////////////////////////
	void Delete2DArray(T **);													// 清除陣列

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
	int NumpyTypeNumber();
};